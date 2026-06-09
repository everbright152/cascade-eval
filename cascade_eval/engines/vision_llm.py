"""Vision-LLM adapter (Claude) — the fallback for low-resource / handwritten pages.

Why it's here: traditional OCR (Tesseract) and the neural detector (EasyOCR)
both fail hardest exactly where we care most — handwriting and low-resource
scripts like Coptic, where EasyOCR has no model at all. A vision LLM reads
those far better, giving the cascade a genuine third option.

The honest catch, and a first-class design point of this harness:
**the Claude API exposes no logprobs or per-token confidence.** The model
returns text with no calibrated certainty signal. So we deliberately set
`confidence = None` here. The router must NOT trust this engine on a
self-reported number (there isn't one) — it judges the LLM candidate by
reference-free proxies (dictionary hit-rate, etc.). This is the cleanest
example in the whole project of "confident vs. correct": an engine that is
fluent and plausible by construction, yet carries zero usable confidence.

Degrades gracefully: if `anthropic` isn't installed or no ANTHROPIC_API_KEY is
set, `is_available` is False and `recognize` returns an EngineResult with
`available=False` — the absence is logged, never crashed.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import os
import time

from ..schemas import EngineName, EngineResult
from .base import Engine, ImageInput, load_dotenv

PROMPT = (
    "You are a careful OCR/HTR transcriber. Transcribe ALL text in this image "
    "exactly as written, preserving original line breaks. Do not translate, "
    "explain, correct spelling, or add commentary. If a region is illegible, "
    "write [illegible] in its place. Output only the transcription."
)


class VisionLLMEngine(Engine):
    name = EngineName.vision_llm

    def __init__(self, model: str = "claude-opus-4-8", max_tokens: int = 4096):
        load_dotenv()  # populate ANTHROPIC_API_KEY from .env if present
        self.model = model
        self.max_tokens = max_tokens

    @property
    def is_available(self) -> bool:
        return (
            importlib.util.find_spec("anthropic") is not None
            and bool(os.environ.get("ANTHROPIC_API_KEY"))
        )

    def for_script(self, script: str) -> "VisionLLMEngine":
        return self  # script-agnostic; the LLM handles all scripts

    @staticmethod
    def _encode(image: ImageInput) -> tuple[str, str]:
        """Return (base64_data, media_type)."""
        import numpy as np

        if isinstance(image, np.ndarray):
            from PIL import Image

            buf = io.BytesIO()
            Image.fromarray(image).save(buf, format="PNG")
            return base64.standard_b64encode(buf.getvalue()).decode("utf-8"), "image/png"

        # path-like
        with open(image, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")
        ext = str(image).lower().rsplit(".", 1)[-1]
        media_type = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        return data, media_type

    def recognize(self, image: ImageInput, page: str) -> EngineResult:
        if not self.is_available:
            return self._unavailable(
                page, "anthropic not installed or ANTHROPIC_API_KEY not set"
            )

        import anthropic

        data, media_type = self._encode(image)
        client = anthropic.Anthropic()
        t0 = time.perf_counter()
        try:
            resp = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": data,
                                },
                            },
                            {"type": "text", "text": PROMPT},
                        ],
                    }
                ],
            )
        except Exception as exc:  # network / auth / rate-limit — log, don't crash
            return self._unavailable(page, f"vision-llm call failed: {type(exc).__name__}: {exc}")
        runtime = time.perf_counter() - t0

        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return EngineResult(
            page=page,
            engine=self.name,
            text=text,
            confidence=None,  # deliberate: API provides no calibrated confidence
            tokens=[],
            runtime_s=runtime,
            available=True,
            note=f"model={self.model}; no per-token confidence available from API",
        )
