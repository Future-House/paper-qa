from pathlib import Path

CACHE_PATH = Path.home() / ".paperqa" / "llm_cache.db"
OCR_CACHE_PATH = CACHE_PATH.parent / "ocr_cache.db"
