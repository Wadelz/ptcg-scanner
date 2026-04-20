from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import Any, Optional
from urllib.parse import urlencode

try:
    from PIL import Image as PILImage
    from PIL.Image import Resampling
except ImportError:  # pragma: no cover - keep query utilities importable without Pillow
    PILImage = None
    Resampling = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - keep query utilities usable without OCR deps
    np = None

try:
    import cv2
except ImportError:  # pragma: no cover - fallback path is covered without OpenCV runtime
    cv2 = None

try:
    import pytesseract
except ImportError:  # pragma: no cover - handled gracefully when dependency is missing
    pytesseract = None

NAME_REGION_HEIGHT_FRACTION = 0.18
NUMBER_REGION_TOP_FRACTION = 0.85
NUMBER_REGION_WIDTH_FRACTION = 0.45
ADAPTIVE_THRESHOLD_BLOCK_SIZE = 31
ADAPTIVE_THRESHOLD_C = 2
PIL_BINARIZATION_THRESHOLD = 160
MAX_COLLECTOR_SEGMENT_LENGTH = 5


@dataclass(frozen=True)
class CardTextExtraction:
    """Structured OCR output for a single detected card crop."""

    card_name: Optional[str]
    collector_number: Optional[str]
    set_total: Optional[str]
    raw_name_text: str
    raw_number_text: str


@dataclass(frozen=True)
class SearchQueryResult:
    """Search query payload used by UI links."""

    query: str
    url: str


class PriceSearchProvider(ABC):
    """Swappable interface for providers that convert extracted data into a lookup URL."""

    @abstractmethod
    def build_query(self, details: CardTextExtraction) -> str:
        raise NotImplementedError

    @abstractmethod
    def build_url(self, query: str) -> str:
        raise NotImplementedError


class WebSearchPriceProvider(PriceSearchProvider):
    """Google/Bing web search implementation for card pricing lookups."""

    def __init__(self, engine: str = "google") -> None:
        self.engine = engine.lower()
        self._base_urls = {
            "google": "https://www.google.com/search",
            "bing": "https://www.bing.com/search",
        }

    def build_query(self, details: CardTextExtraction) -> str:
        query_parts = []
        if details.card_name:
            query_parts.append(f"\"{details.card_name}\"")
        if details.collector_number and details.set_total:
            query_parts.append(f"\"{details.collector_number}/{details.set_total}\"")
        elif details.collector_number:
            query_parts.append(f"\"{details.collector_number}\"")
        query_parts.append("pokemon card price")
        return " ".join(query_parts)

    def build_url(self, query: str) -> str:
        base_url = self._base_urls.get(self.engine, self._base_urls["google"])
        return f"{base_url}?{urlencode({'q': query})}"


def extract_card_text_details(card_crop: Any) -> CardTextExtraction:
    """
    Extract card name and collector number details from key regions in a card crop.

    Regions:
    - Name: top ~18% of full width
    - Collector number: bottom-left (~15% height, ~45% width)
    """

    if PILImage is None:
        return CardTextExtraction(
            card_name=None,
            collector_number=None,
            set_total=None,
            raw_name_text="",
            raw_number_text="",
        )
    if not isinstance(card_crop, PILImage.Image):
        raise TypeError("card_crop must be a PIL.Image.Image instance")

    width, height = card_crop.size
    name_region = card_crop.crop((0, 0, width, max(1, int(height * NAME_REGION_HEIGHT_FRACTION))))
    number_region = card_crop.crop(
        (
            0,
            max(0, int(height * NUMBER_REGION_TOP_FRACTION)),
            max(1, int(width * NUMBER_REGION_WIDTH_FRACTION)),
            height,
        )
    )

    raw_name_text = _run_ocr(name_region, tesseract_config="--oem 3 --psm 7")
    raw_number_text = _run_ocr(
        number_region,
        tesseract_config="--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/",
    )

    cleaned_name = _clean_name(raw_name_text)
    collector_number, set_total = _extract_collector_fraction(raw_number_text)

    return CardTextExtraction(
        card_name=cleaned_name,
        collector_number=collector_number,
        set_total=set_total,
        raw_name_text=raw_name_text,
        raw_number_text=raw_number_text,
    )


def build_price_search_query(details: CardTextExtraction, provider: Optional[PriceSearchProvider] = None) -> SearchQueryResult:
    """Build deterministic search query + URL for pricing lookup."""

    selected_provider = provider or WebSearchPriceProvider(engine="google")
    query = selected_provider.build_query(details)
    return SearchQueryResult(query=query, url=selected_provider.build_url(query))


def _run_ocr(region_image: Any, tesseract_config: str) -> str:
    """Run OCR with a tesseract config string, returning an empty string on OCR/runtime failures."""

    if pytesseract is None:
        return ""

    processed = _preprocess_for_ocr(region_image)
    try:
        return pytesseract.image_to_string(processed, config=tesseract_config).strip()
    except (RuntimeError, OSError, ValueError):
        return ""


def _preprocess_for_ocr(region_image: Any) -> Any:
    """
    OCR-focused preprocessing: grayscale, upscale, denoise and threshold.

    Returns:
        np.ndarray: thresholded image when OpenCV+NumPy are available.
        PIL.Image.Image: thresholded grayscale image for fallback processing.
    """

    if cv2 is not None and np is not None:
        region_rgb = np.array(region_image.convert("RGB"))
        gray = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2GRAY)
        upscaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        denoised = cv2.GaussianBlur(upscaled, (3, 3), 0)
        return cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            ADAPTIVE_THRESHOLD_BLOCK_SIZE,
            ADAPTIVE_THRESHOLD_C,
        )

    if Resampling is None:
        return region_image

    pil_gray = region_image.convert("L").resize(
        (region_image.size[0] * 2, region_image.size[1] * 2),
        Resampling.BICUBIC,
    )
    return pil_gray.point(lambda value: 255 if value > PIL_BINARIZATION_THRESHOLD else 0)


def _clean_name(raw_name_text: str) -> Optional[str]:
    """Normalize OCR name output by collapsing whitespace and trimming quote-like wrappers."""

    name = re.sub(r"\s+", " ", raw_name_text).strip(" \"'\n\t")
    return name or None


def _extract_collector_fraction(raw_number_text: str) -> tuple[Optional[str], Optional[str]]:
    text = raw_number_text.replace(" ", "")
    match = re.search(
        rf"([A-Za-z0-9]{{1,{MAX_COLLECTOR_SEGMENT_LENGTH}}})/([A-Za-z0-9]{{1,{MAX_COLLECTOR_SEGMENT_LENGTH}}})",
        text,
    )
    if not match:
        return None, None
    return match.group(1), match.group(2)
