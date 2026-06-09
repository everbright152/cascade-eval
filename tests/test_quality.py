"""Step 4 unit checks: script detection + reference-free proxies behave sanely
on clean vs. garbage, and honestly report when a signal can't be computed.

Run:
    python -m tests.test_quality
    pytest tests/test_quality.py
"""

from __future__ import annotations

import numpy as np

from cascade_eval.preprocess import preprocess
from cascade_eval.quality import (
    assess,
    dictionary_hit_rate,
    script_consistency,
)
from cascade_eval.schemas import EngineName, EngineResult
from cascade_eval.script_detect import detect_script

ARABIC = "مرحبا بالعالم"
LATIN = "hello world"
COPTIC = "ⲡⲣⲱⲙⲉ ⲛⲟⲩⲧⲉ"


def test_script_detection():
    assert detect_script(ARABIC).dominant == "Arabic"
    assert detect_script(ARABIC).is_rtl is True
    assert detect_script(LATIN).dominant == "Latin"
    assert detect_script(LATIN).is_rtl is False
    assert detect_script(COPTIC).dominant == "Coptic"
    assert detect_script("12345 .,!").dominant is None  # no script-bearing chars


def test_script_consistency_catches_wrong_script():
    # Right script -> high; wrong script (fluent gibberish) -> low.
    assert script_consistency(ARABIC, "Arabic") == 1.0
    assert script_consistency(LATIN, "Arabic") == 0.0
    # No script chars at all -> cannot compute.
    assert script_consistency("123 ...", "Arabic") is None


def test_dictionary_hit_rate():
    wl = {"hello", "world"}
    assert dictionary_hit_rate("hello world", wl) == 1.0
    assert dictionary_hit_rate("hello xyzzy", wl) == 0.5
    assert dictionary_hit_rate("xyzzy plugh", wl) == 0.0
    # No wordlist (low-resource language) -> None, NOT a fabricated 0.
    assert dictionary_hit_rate("anything", None) is None


def test_assess_handles_llm_no_confidence():
    # vision-LLM has confidence=None; proxy must still be computable via script.
    llm = EngineResult(page="cop_001", engine=EngineName.vision_llm, text=COPTIC, confidence=None)
    rep = assess(llm, expected_script="Coptic", wordlist=None)
    assert rep.engine_confidence is None          # honestly absent
    assert rep.dictionary_hit is None             # no Coptic wordlist
    assert rep.script_consistency == 1.0          # but this still works
    assert rep.combined() == 1.0                  # combined from the one available signal


def test_assess_confident_but_wrong_script():
    # An engine confident (0.95) yet emitting the wrong script -> low combined.
    bad = EngineResult(page="ara_001", engine=EngineName.tesseract, text=LATIN, confidence=0.95)
    rep = assess(bad, expected_script="Arabic", wordlist=None)
    assert rep.engine_confidence == 0.95
    assert rep.script_consistency == 0.0
    assert rep.combined() < 0.5  # the confidence is real, the output is not


def test_preprocess_shape():
    arr = (np.random.default_rng(0).integers(0, 256, (40, 60, 3))).astype(np.uint8)
    out = preprocess(arr, do_deskew=False)
    assert out.shape == (40, 60, 3)
    assert out.dtype == np.uint8


if __name__ == "__main__":
    for fn in [
        test_script_detection,
        test_script_consistency_catches_wrong_script,
        test_dictionary_hit_rate,
        test_assess_handles_llm_no_confidence,
        test_assess_confident_but_wrong_script,
        test_preprocess_shape,
    ]:
        fn()
        print(f"OK: {fn.__name__}")
    print("\nAll Step 4 signal checks passed.")
