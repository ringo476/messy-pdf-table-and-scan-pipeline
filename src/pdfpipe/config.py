"""All settings live here. Any of them can be overridden with a PDFPIPE_* environment variable."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PDFPIPE_", env_file=".env", extra="ignore")

    # Upload limits (step 1)
    max_upload_mb: int = 50
    max_pages: int = 200
    max_page_side_points: float = 5000.0

    # Rendering (step 3)
    digital_dpi: int = 144
    scan_dpi: int = 216
    max_render_pixels: int = 25_000_000

    # Page classification (step 2)
    min_text_words: int = 3
    garbled_text_limit: float = 0.6
    full_page_image_cover: float = 0.7
    mixed_image_cover: float = 0.03
    text_layer_min_agreement: float = 0.6

    # Models (steps 5, 6, 9)
    layout_min_score: float = 0.35
    model_threads: int = 2

    # Missed-text check (step 7): how many text-like blobs with no word on them trigger full-page OCR
    missed_text_min_blobs: int = 4

    # Scoring (step 15)
    min_ocr_confidence: float = 0.80
    min_field_score: float = 0.85
    money_tolerance: float = 0.01

    # Claude (step 16): off unless explicitly enabled and a key is present
    llm_enabled: bool = False
    anthropic_api_key: str | None = None
    llm_first_model: str = "claude-haiku-5-5"
    llm_second_model: str = "claude-sonnet-5-5"
    llm_max_tokens: int = 16000  # A ceiling, not a cost: only tokens actually written are billed.
    # Spending limits. Once a limit is reached, the remaining failed fields simply go to a person.
    llm_max_cost_per_document_usd: float = 0.25
    llm_daily_budget_usd: float = 5.0
    # After this many failed calls in a row, this worker stops calling Claude for a while.
    llm_pause_after_failures: int = 5
    llm_pause_seconds: int = 300

    # Service
    database_url: str = "postgresql://pdfpipe:pdfpipe@localhost:5432/pdfpipe"
    data_dir: Path = Path("data")
    session_hours: int = 12
    cookie_secure: bool = False
    max_active_jobs_per_user: int = 20
    job_timeout_seconds: int = 900
    job_max_attempts: int = 3
    # A worker renews its lease every 5 seconds, so a dead worker's job is picked up again after this long.
    job_lease_seconds: int = 20
