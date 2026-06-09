"""EasyOCR adapter — the deep-learning contrast to Tesseract.

Included because it fails *differently* from Tesseract (neural detector +
recognizer vs. classical pipeline), which is what makes a routing decision
between them meaningful rather than redundant. Returns per-box confidence
already in [0,1]. Readers are heavy to construct, so they're built lazily and
cached per language set.
"""

from __future__ import annotations

import importlib.util
import time

from ..schemas import BBox, EngineName, EngineResult, TokenConfidence
from .base import Engine, ImageInput

# EasyOCR language codes. Note: EasyOCR has NO Coptic model — that gap is real
# and is exactly the kind of thin coverage the cascade must route around.
SCRIPT_TO_LANGS = {
    "Latin": ["en"],
    "Arabic": ["ar"],
    "Jawi": ["ar"],
    # "Coptic": unsupported -> handled as unavailable for this script
}

_READER_CACHE: dict = {}


class EasyOCREngine(Engine):
    name = EngineName.easyocr

    def __init__(self, langs: list[str] | None = None):
        self.langs = langs or ["en"]

    @property
    def is_available(self) -> bool:
        return importlib.util.find_spec("easyocr") is not None

    def for_script(self, script: str) -> "EasyOCREngine | None":
        """Return an adapter for the script, or None if EasyOCR can't do it."""
        langs = SCRIPT_TO_LANGS.get(script)
        if langs is None:
            return None
        return EasyOCREngine(langs=langs)

    def _reader(self):
        key = tuple(self.langs)
        if key not in _READER_CACHE:
            import easyocr

            _READER_CACHE[key] = easyocr.Reader(self.langs, gpu=False)
        return _READER_CACHE[key]

    def recognize(self, image: ImageInput, page: str) -> EngineResult:
        if not self.is_available:
            return self._unavailable(page, "easyocr not installed")

        arr = self._to_ndarray(image)
        t0 = time.perf_counter()
        # detail=1 -> [ (bbox_pts, text, conf), ... ]
        results = self._reader().readtext(arr, detail=1, paragraph=False)
        runtime = time.perf_counter() - t0

        tokens: list[TokenConfidence] = []
        confs: list[float] = []
        words: list[str] = []
        for bbox_pts, txt, conf in results:
            txt = (txt or "").strip()
            if not txt:
                continue
            xs = [int(p[0]) for p in bbox_pts]
            ys = [int(p[1]) for p in bbox_pts]
            words.append(txt)
            confs.append(float(conf))
            tokens.append(
                TokenConfidence(
                    text=txt,
                    confidence=float(conf),
                    bbox=BBox(x=min(xs), y=min(ys), w=max(xs) - min(xs), h=max(ys) - min(ys)),
                )
            )

        text = " ".join(words)
        page_conf = sum(confs) / len(confs) if confs else None
        return EngineResult(
            page=page, engine=self.name, text=text, confidence=page_conf,
            tokens=tokens, runtime_s=runtime, available=True,
            note=f"langs={','.join(self.langs)}",
        )
