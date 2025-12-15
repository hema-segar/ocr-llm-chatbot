import easyocr
import os
import logging
import threading
from typing import List, Dict, Tuple, Any, Optional

LANGUAGES = ["en"]  # Define your default languages here (e.g., ['en', 'fr'])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy-initialized global reader. Initializing EasyOCR can be expensive so we do it on demand.
_READER: Optional[easyocr.Reader] = None
_READER_LOCK = threading.Lock()


def init_reader(languages: List[str] = LANGUAGES, gpu: bool = False, force: bool = False) -> Optional[easyocr.Reader]:
    """
    Initialize and return the global EasyOCR reader. Thread-safe and idempotent unless force=True.
    """
    global _READER
    with _READER_LOCK:
        if _READER is None or force:
            try:
                logger.info("Initializing EasyOCR reader (gpu=%s, langs=%s)...", gpu, languages)
                _READER = easyocr.Reader(languages, gpu=gpu)
                logger.info("EasyOCR reader initialized.")
            except Exception as e:
                _READER = None
                logger.exception("Failed to initialize EasyOCR reader: %s", e)
    return _READER


def get_reader() -> Optional[easyocr.Reader]:
    """Return the global reader, initializing with defaults if necessary."""
    global _READER
    if _READER is None:
        init_reader()
    return _READER


def recognize_text_from_image(image_path: str, reader: Optional[easyocr.Reader] = None) -> List[Dict[str, Any]]:
    """
    Performs OCR on the specified image file and returns structured results.

    Returns a list of dicts: [{'text': str, 'confidence': float, 'bbox': [...]}, ...]
    Returns an empty list on error.
    """
    reader = reader or get_reader()
    if reader is None:
        logger.error("OCR Reader is not available.")
        return []

    if not os.path.exists(image_path):
        logger.error("Image file not found: %s", image_path)
        return []

    try:
        results: List[Tuple[Any, str, float]] = reader.readtext(image_path)
        structured_results: List[Dict[str, Any]] = []
        for (bbox, text, conf) in results:
            structured_results.append({
                "text": text,
                "confidence": conf,
                "bbox": bbox,
            })
        return structured_results
    except Exception as e:
        logger.exception("An error occurred during OCR processing of %s: %s", image_path, e)
        return []


def extract_text_from_image(image_path: str, reader: Optional[easyocr.Reader] = None) -> str:
    """
    Perform OCR on an image and return concatenated plain text suitable for chunking.

    Returns an empty string on error or if nothing is recognized.
    """
    structured = recognize_text_from_image(image_path, reader=reader)
    if not structured:
        return ""
    # Join recognized text blocks in approximate reading order
    texts = [entry.get("text", "") for entry in structured if entry.get("text")]
    return "\n".join(texts)


__all__ = [
    "init_reader",
    "get_reader",
    "recognize_text_from_image",
    "extract_text_from_image",
]
