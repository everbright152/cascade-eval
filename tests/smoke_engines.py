"""Step 3 smoke test: every engine honors the recognize() contract.

Generates a synthetic Latin and Arabic image (no corpus needed), runs each
configured engine, and asserts the uniform contract:
  - returns an EngineResult for the right page/engine
  - if the engine can't run (dep missing / no API key), available=False + a note
  - if it can run, available=True

This passes whether or not the engines are installed — an uninstalled engine
must degrade to a logged result, not an exception. Installed engines actually
transcribe.

Run:
    python -m tests.smoke_engines
    pytest tests/smoke_engines.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from cascade_eval.engines import build_engines
from cascade_eval.schemas import EngineResult

CONFIG = {
    "engines": {
        "tesseract": {"enabled": True},
        "easyocr": {"enabled": True},
        "vision_llm": {"enabled": True, "model": "claude-opus-4-8"},
    }
}


def _make_image(text: str) -> np.ndarray:
    img = Image.new("RGB", (480, 120), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 40), text, fill="black")  # default bitmap font; fine for smoke
    return np.array(img)


def test_engine_contract():
    images = {"lat_smoke": _make_image("Hello World"), "ara_smoke": _make_image("Test 123")}
    engines = build_engines(CONFIG)
    assert engines, "no engines were built"

    for name, engine in engines.items():
        for page, arr in images.items():
            result = engine.recognize(arr, page)
            assert isinstance(result, EngineResult)
            assert result.engine == name
            assert result.page == page
            if result.available:
                assert isinstance(result.text, str)
            else:
                assert result.note, f"{name} unavailable but gave no reason"


def _report():
    """Human-readable status of which engines are live on this machine."""
    engines = build_engines(CONFIG)
    img = _make_image("Hello World")
    print("engine            available  result")
    print("-" * 60)
    with tempfile.TemporaryDirectory():
        for name, engine in engines.items():
            res = engine.recognize(img, "lat_smoke")
            status = "yes" if res.available else "no "
            detail = (
                f'text={res.text[:30]!r} conf={res.confidence}'
                if res.available
                else f"({res.note})"
            )
            print(f"{name.value:<17} {status:<10} {detail}")


if __name__ == "__main__":
    test_engine_contract()
    print("OK: all engines honor the recognize() contract\n")
    _report()
