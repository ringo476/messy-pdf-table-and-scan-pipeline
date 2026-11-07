"""The three local models. Each loads on first use, and one worker keeps one copy of each."""

import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from .config import Settings
from .models import BBox, Word, WordSource

logger = logging.getLogger(__name__)


@dataclass
class DetectedRegion:
    label: str
    bbox: BBox
    score: float


@dataclass
class DetectedCell:
    bbox: BBox
    row: int
    col: int
    rowspan: int
    colspan: int


class Engines:
    """Wraps the model libraries, so the rest of the code only sees plain boxes and words."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @cached_property
    def _ocr(self) -> Any:
        from rapidocr import RapidOCR

        return RapidOCR(
            params={
                "Global.log_level": "error",
                # Never enlarge a small crop. Enlarging it makes the detector split lines and re-read
                # letters, which produced fragments like "Orbit t Systems".
                "Det.limit_type": "max",
                "Det.limit_side_len": 2000,
                "EngineConfig.onnxruntime.intra_op_num_threads": self.settings.model_threads,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            }
        )

    @cached_property
    def _layout(self) -> Any:
        from rapid_layout import ModelType, RapidLayout, RapidLayoutInput

        return RapidLayout(
            RapidLayoutInput(
                model_type=ModelType.PP_DOC_LAYOUTV3,
                conf_thresh=self.settings.layout_min_score,
                engine_cfg={
                    "intra_op_num_threads": self.settings.model_threads,
                    "inter_op_num_threads": 1,
                },
            )
        )

    @cached_property
    def _table(self) -> Any:
        from rapid_table import RapidTable, RapidTableInput

        return RapidTable(
            RapidTableInput(
                use_ocr=False,
                engine_cfg={
                    "intra_op_num_threads": self.settings.model_threads,
                    "inter_op_num_threads": 1,
                },
            )
        )

    def warm_up(self) -> None:
        """Loads every model now, so a worker's first job isn't slowed down by it."""
        blank = np.full((200, 400, 3), 255, dtype=np.uint8)
        self.read_words(blank)
        self.find_regions(blank)
        self._table  # noqa: B018

    def find_regions(self, image: np.ndarray) -> list[DetectedRegion]:
        output = self._layout(image)
        if output.boxes is None or len(output.boxes) == 0:
            return []
        return [
            DetectedRegion(
                label=str(label),
                bbox=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                score=float(score),
            )
            for b, label, score in zip(output.boxes, output.class_names, output.scores, strict=True)
        ]

    def read_words(self, image: np.ndarray, offset: tuple[float, float] = (0.0, 0.0)) -> list[Word]:
        """Runs OCR on an image. Positions are shifted by offset, so crops map back to the page."""
        if image.size == 0:
            return []
        # Page orientation is fixed earlier, so the per-line rotation guess is switched off.
        output = self._ocr(image, use_cls=False, return_word_box=True)
        lines = getattr(output, "word_results", None)
        if not lines:
            return []
        words: list[Word] = []
        for line in lines:
            for entry in line:
                # A blank image gives placeholder values instead of (text, score, box) entries.
                if not isinstance(entry, tuple | list) or len(entry) != 3:
                    continue
                text, score, box = entry
                cleaned = str(text).strip()
                if not cleaned:
                    continue
                xs = [float(point[0]) for point in box]
                ys = [float(point[1]) for point in box]
                words.append(
                    Word(
                        text=cleaned,
                        bbox=(
                            min(xs) + offset[0],
                            min(ys) + offset[1],
                            max(xs) + offset[0],
                            max(ys) + offset[1],
                        ),
                        source=WordSource.OCR,
                        confidence=float(score),
                    )
                )
        return words

    def find_cells(self, crop: np.ndarray, words: list[Word], offset: tuple[float, float]) -> list[DetectedCell]:
        """Finds rows, columns and merged cells in a table image. Positions come back in page pixels."""
        if words:
            boxes = np.array(
                [
                    [
                        [w.bbox[0] - offset[0], w.bbox[1] - offset[1]],
                        [w.bbox[2] - offset[0], w.bbox[1] - offset[1]],
                        [w.bbox[2] - offset[0], w.bbox[3] - offset[1]],
                        [w.bbox[0] - offset[0], w.bbox[3] - offset[1]],
                    ]
                    for w in words
                ],
                dtype=np.float32,
            )
            texts = tuple(w.text for w in words)
            scores = tuple(1.0 if w.confidence is None else w.confidence for w in words)
        else:
            boxes = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]], dtype=np.float32)
            texts, scores = ("",), (1.0,)
        output = self._table(crop, ocr_results=[(boxes, texts, scores)])
        cells: list[DetectedCell] = []
        for polygon, logic in zip(output.cell_bboxes[0], output.logic_points[0], strict=True):
            xs = [float(v) for v in polygon[0::2]]
            ys = [float(v) for v in polygon[1::2]]
            row_start, row_end, col_start, col_end = (int(v) for v in logic)
            cells.append(
                DetectedCell(
                    bbox=(min(xs) + offset[0], min(ys) + offset[1], max(xs) + offset[0], max(ys) + offset[1]),
                    row=row_start,
                    col=col_start,
                    rowspan=max(1, row_end - row_start + 1),
                    colspan=max(1, col_end - col_start + 1),
                )
            )
        return cells
