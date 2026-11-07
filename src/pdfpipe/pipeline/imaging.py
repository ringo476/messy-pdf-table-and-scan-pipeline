"""Steps 3 and 4: convert a page to an image, and clean up scanned images."""

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..config import Settings
from ..engines import Engines
from ..models import Word


def render_page(pdfium_page: Any, dpi: int, settings: Settings) -> tuple[np.ndarray, float]:
    """Step 3. Returns the page as a BGR image, plus the pixels-per-point scale actually used."""
    width, height = pdfium_page.get_size()
    scale = dpi / 72
    pixels = width * scale * height * scale
    if pixels > settings.max_render_pixels:
        scale *= (settings.max_render_pixels / pixels) ** 0.5
    bitmap = pdfium_page.render(scale=scale)
    try:
        image = bitmap.to_numpy()
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image.copy(), scale
    finally:
        bitmap.close()


@dataclass
class CleanedImage:
    image: np.ndarray
    rotated_degrees: int = 0
    deskew_degrees: float = 0.0
    notes: list[str] = field(default_factory=list)


def _rotate_white(image: np.ndarray, degrees: float) -> np.ndarray:
    """Rotates an image, growing the canvas so no corner is cut off."""
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width, new_height = int(height * sin + width * cos), int(height * cos + width * sin)
    matrix[0, 2] += new_width / 2 - width / 2
    matrix[1, 2] += new_height / 2 - height / 2
    return cv2.warpAffine(image, matrix, (new_width, new_height), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def estimate_skew(image: np.ndarray, max_degrees: float = 12.0) -> float:
    """Finds the rotation that makes text lines most sharply horizontal.

    Straight text lines give a row-by-row ink count with tall peaks and empty gaps, so the angle with
    the highest variance in that count is the correct one.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 900 / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, ink = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.count_nonzero(ink) < 50:
        return 0.0
    height, width = ink.shape
    center = (width / 2, height / 2)

    def sharpness(angle: float) -> float:
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        turned = cv2.warpAffine(ink, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
        return float(np.var(turned.sum(axis=1, dtype=np.float64)))

    coarse = np.arange(-max_degrees, max_degrees + 0.01, 0.5)
    best = float(coarse[int(np.argmax([sharpness(float(a)) for a in coarse]))])
    fine = np.arange(best - 0.5, best + 0.51, 0.1)
    return round(float(fine[int(np.argmax([sharpness(float(a)) for a in fine]))]), 1)


def estimate_noise(gray: np.ndarray) -> float:
    """A standard estimate of pixel noise, based on the median response of an edge filter."""
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.median(np.abs(laplacian)) / 0.6745 / 2.45)


def _reading_score(words: list[Word]) -> float:
    return sum(len(w.text) * (w.confidence or 0.0) ** 2 for w in words)


def fix_orientation(image: np.ndarray, engines: Engines) -> tuple[np.ndarray, int]:
    """Detects a page that is sideways or upside down by reading a small copy of it each way."""
    scale = min(1.0, 1100 / max(image.shape[:2]))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    upright = engines.read_words(small)
    rotations = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }

    # OCR can read sideways text, so good reading alone doesn't prove a page is upright.
    # On a sideways page, most words are taller than they are wide.
    longer = [w for w in upright if len(w.text) >= 3]
    tall = [w for w in longer if (w.bbox[3] - w.bbox[1]) > (w.bbox[2] - w.bbox[0])]
    if longer and len(tall) > 0.5 * len(longer):
        best_degrees = max(
            (90, 270), key=lambda degrees: _reading_score(engines.read_words(cv2.rotate(small, rotations[degrees])))
        )
        return cv2.rotate(image, rotations[best_degrees]), best_degrees

    confident = [w for w in upright if (w.confidence or 0.0) >= 0.9]
    # A horizontal page that reads well needs no further passes, which saves OCR runs.
    if len(confident) >= 15 and len(confident) >= 0.7 * len(upright):
        return image, 0

    # Poor reading on a horizontal page may mean it is upside down. Rotating must be clearly
    # better, because noise alone shouldn't turn a page.
    upside_down = _reading_score(engines.read_words(cv2.rotate(small, rotations[180])))
    if upside_down > _reading_score(upright) * 1.3 and upside_down > 20:
        return cv2.rotate(image, rotations[180]), 180
    return image, 0


def clean_scan(image: np.ndarray, engines: Engines) -> CleanedImage:
    """Step 4. Turns the page upright, straightens it, reduces noise and fixes faded contrast.

    The image is not converted to pure black and white, because the OCR and layout models read
    grey levels better than a hard threshold on poor scans.
    """
    notes: list[str] = []
    image, rotated = fix_orientation(image, engines)
    if rotated:
        notes.append(f"rotated page by {rotated} degrees")

    skew = estimate_skew(image)
    if abs(skew) >= 0.2:
        image = _rotate_white(image, skew)
        notes.append(f"straightened page by {skew} degrees")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noise = estimate_noise(gray)
    if noise > 6:
        image = cv2.medianBlur(image, 3)
        notes.append(f"reduced noise (estimated level {noise:.1f})")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Text is a tiny share of a page, so the dark end is measured far into the tail.
    low, high = np.percentile(gray, [0.2, 99])
    if high - low < 170 and high > low:
        stretched = np.clip((image.astype(np.float32) - low) * 255.0 / (high - low), 0, 255)
        image = stretched.astype(np.uint8)
        notes.append("raised faded contrast")

    return CleanedImage(
        image=image, rotated_degrees=rotated, deskew_degrees=skew if abs(skew) >= 0.2 else 0.0, notes=notes
    )
