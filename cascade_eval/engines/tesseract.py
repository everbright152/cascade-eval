"""Tesseract adapter — the baseline engine.

Chosen as the baseline for its breadth of script coverage (Arabic, Coptic,
Latin via traineddata) and, crucially, because `image_to_data` exposes
*per-word confidence*. That per-token signal is what the router and the
confidence-vs-correctness analysis are built on.
"""

from __future__ import annotations

import importlib.util
import time

from ..schemas import BBox, EngineName, EngineResult, TokenConfidence
from .base import Engine, ImageInput

# Map our script labels -> Tesseract language/traineddata codes.
SCRIPT_TO_LANG = {
    "Latin": "eng",
    "Arabic": "ara",
    "Coptic": "cop",
    "Jawi": "ara",  # Jawi uses Arabic-derived script; closest available pack
}


class TesseractEngine(Engine):
    name = EngineName.tesseract

    def __init__(self, lang: str = "eng"):
        self.lang = lang

    @property
    def is_available(self) -> bool:
        if importlib.util.find_spec("pytesseract") is None:
            return False
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False  # binary not on PATH

    def for_script(self, script: str) -> "TesseractEngine":
        """Return an adapter configured for the page's script."""
        return TesseractEngine(lang=SCRIPT_TO_LANG.get(script, "eng"))

    def recognize(self, image: ImageInput, page: str) -> EngineResult:
        if not self.is_available:
            return self._unavailable(page, "pytesseract or tesseract binary not available")

        import pytesseract
        from pytesseract import Output

        pil = self._to_pil(image)
        t0 = time.perf_counter()
        data = pytesseract.image_to_data(pil, lang=self.lang, output_type=Output.DICT)
        runtime = time.perf_counter() - t0

        tokens: list[TokenConfidence] = []
        confs: list[float] = []
        words: list[str] = []
        for i, word in enumerate(data["text"]):
            word = word.strip()
            conf_raw = float(data["conf"][i])
            if not word or conf_raw < 0:  # tesseract uses -1 for non-text boxes
                continue
            conf = conf_raw / 100.0  # tesseract conf is 0-100 -> normalize to [0,1]
            words.append(word)
            confs.append(conf)
            tokens.append(
                TokenConfidence(
                    text=word,
                    confidence=conf,
                    bbox=BBox(
                        x=data["left"][i], y=data["top"][i],
                        w=data["width"][i], h=data["height"][i],
                    ),
                )
            )

        text = " ".join(words)
        page_conf = sum(confs) / len(confs) if confs else None
        return EngineResult(
            page=page, engine=self.name, text=text, confidence=page_conf,
            tokens=tokens, runtime_s=runtime, available=True,
            note=f"lang={self.lang}",
        )
