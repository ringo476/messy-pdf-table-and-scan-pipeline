"""The language models that can read an invoice's fields, behind one small interface.

A reader takes a system prompt, some text and page images, and a JSON schema, and returns what the model said as
a dict. The rest of the pipeline never knows which company's model answered. That keeps the checks, the spending
limits and the "never trust the model on its own" rules in one place, and makes a provider easy to swap.

A reader only reads. It can't call tools, change a decision, or take any action.
"""

import base64
import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from ..config import Settings

logger = logging.getLogger(__name__)

# A part of the request: ("text", str) or ("image", JPEG bytes).
Part = tuple[str, Any]


@dataclass
class Reply:
    """What came back. data is None when the answer was refused, cut off, or not the requested JSON."""

    data: dict[str, Any] | None
    input_tokens: int = 0
    output_tokens: int = 0
    served_by: str = ""


class ReaderError(Exception):
    """The call failed, with no connection or an error from the service. The caller counts these to pause calls."""


class Reader(Protocol):
    def ask(self, model: str, system: str, parts: list[Part], schema: dict[str, Any], max_tokens: int) -> Reply: ...


def models_for(settings: Settings) -> tuple[str, str | None]:
    """The first and the optional second model for the chosen provider."""
    if settings.llm_provider == "anthropic":
        return settings.llm_first_model, settings.llm_second_model
    return settings.openai_first_model, settings.openai_second_model


def _parse(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("the model's answer was not valid JSON, so it was ignored")
        return None
    return parsed if isinstance(parsed, dict) else None


class OpenAIReader:
    """Chat Completions with a strict JSON schema. Images go in as base64 data, never as a link."""

    def __init__(self, settings: Settings, client: Any = None) -> None:
        if client is None:
            import openai

            client = openai.OpenAI(api_key=settings.openai_api_key, timeout=90.0, max_retries=2)
        self.client = client
        self.detail = settings.openai_image_detail

    def ask(self, model: str, system: str, parts: list[Part], schema: dict[str, Any], max_tokens: int) -> Reply:
        import openai

        content: list[dict[str, Any]] = []
        for kind, value in parts:
            if kind == "image":
                data = base64.standard_b64encode(value).decode("ascii")
                content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}", "detail": self.detail}}
                )
            else:
                content.append({"type": "text", "text": value})
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "invoice_fields", "strict": True, "schema": schema},
                },
                max_completion_tokens=max_tokens,
            )
        except openai.APIStatusError as error:
            raise ReaderError(f"status {error.status_code}") from error
        except openai.APIConnectionError as error:  # This includes timeouts.
            raise ReaderError("could not connect") from error

        usage = getattr(response, "usage", None)
        reply = Reply(
            data=None,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            served_by=str(getattr(response, "model", "") or model),
        )
        choice = response.choices[0]
        if choice.finish_reason != "stop" or getattr(choice.message, "refusal", None):
            logger.warning("the model stopped with %s, so its answer was ignored", choice.finish_reason)
            return reply
        reply.data = _parse(choice.message.content)
        return reply


class AnthropicReader:
    """The same interface over Claude. It is only used when the provider is set to anthropic."""

    def __init__(self, settings: Settings, client: Any = None) -> None:
        if client is None:
            import anthropic

            # With no key given, the SDK finds other credentials, such as an `ant auth login` profile.
            options: dict[str, Any] = {"timeout": 90.0, "max_retries": 2}
            if settings.anthropic_api_key:
                options["api_key"] = settings.anthropic_api_key
            client = anthropic.Anthropic(**options)
        self.client = client

    def ask(self, model: str, system: str, parts: list[Part], schema: dict[str, Any], max_tokens: int) -> Reply:
        import anthropic

        content: list[dict[str, Any]] = []
        for kind, value in parts:
            if kind == "image":
                data = base64.standard_b64encode(value).decode("ascii")
                content.append(
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}
                )
            else:
                content.append({"type": "text", "text": value})
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": content}],
            "output_config": {
                # Reading printed values doesn't need deep reasoning, which keeps the cost low.
                "effort": "low",
                "format": {"type": "json_schema", "schema": schema},
            },
        }
        try:
            if model.startswith("claude-sonnet"):
                # If a safety classifier wrongly declines an invoice, the API retries on another model.
                response = self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request
                )
            else:
                response = self.client.messages.create(**request)
        except anthropic.APIStatusError as error:
            raise ReaderError(f"status {error.status_code}") from error
        except anthropic.APIConnectionError as error:
            raise ReaderError("could not connect") from error

        usage = getattr(response, "usage", None)
        reply = Reply(
            data=None,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            served_by=getattr(response, "model", model) or model,
        )
        if response.stop_reason != "end_turn":
            logger.warning("the model stopped with %s, so its answer was ignored", response.stop_reason)
            return reply
        reply.data = _parse(next((block.text for block in response.content if block.type == "text"), None))
        return reply


def build_reader(settings: Settings) -> Reader | None:
    """The reader for the chosen provider, or None if the language model is off or has no key."""
    if not settings.llm_enabled:
        return None
    if settings.llm_provider == "anthropic":
        return AnthropicReader(settings)
    if settings.llm_provider != "openai":
        logger.error("PDFPIPE_LLM_PROVIDER must be 'openai' or 'anthropic', so the language model stays off")
        return None
    if not settings.openai_api_key:
        logger.error("the language model is turned on but PDFPIPE_OPENAI_API_KEY is empty, so it stays off")
        return None
    return OpenAIReader(settings)
