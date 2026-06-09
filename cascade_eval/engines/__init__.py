"""OCR/HTR engine adapters. Each exposes a uniform recognize() -> EngineResult."""

from __future__ import annotations

from ..schemas import EngineName
from .base import Engine
from .easyocr import EasyOCREngine
from .tesseract import TesseractEngine
from .vision_llm import VisionLLMEngine

__all__ = ["Engine", "TesseractEngine", "EasyOCREngine", "VisionLLMEngine", "build_engines"]


def build_engines(config: dict, allow_llm: bool = True) -> dict[EngineName, Engine]:
    """Instantiate the engines enabled in config. Construction is cheap and does
    not load models or hit the network — availability is checked lazily."""
    engines_cfg = config.get("engines", {})
    engines: dict[EngineName, Engine] = {}

    if engines_cfg.get("tesseract", {}).get("enabled", True):
        engines[EngineName.tesseract] = TesseractEngine()
    if engines_cfg.get("easyocr", {}).get("enabled", True):
        engines[EngineName.easyocr] = EasyOCREngine()
    if allow_llm and engines_cfg.get("vision_llm", {}).get("enabled", True):
        model = engines_cfg.get("vision_llm", {}).get("model", "claude-opus-4-8")
        engines[EngineName.vision_llm] = VisionLLMEngine(model=model)

    return engines
