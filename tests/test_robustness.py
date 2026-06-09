"""Step 7: perturbations produce valid images, and the robustness slice flags
the silent-failure case (CER up, confidence flat) — distinct from an honest
failure (CER up, confidence drops) and the no-confidence LLM case.

Run:
    python -m tests.test_robustness
    pytest tests/test_robustness.py
"""

from __future__ import annotations

import numpy as np

from cascade_eval.metrics import PageScore
from cascade_eval.perturb import apply_perturbations
from cascade_eval.robustness import compute_robustness
from cascade_eval.schemas import EngineName

CONFIG = {
    "robustness": {
        "perturbations": {"blur": {"kernel": 5}, "skew": {"degrees": 7}, "lowres": {"scale": 0.5}}
    }
}


def _img():
    return (np.random.default_rng(1).integers(0, 256, (50, 80, 3))).astype(np.uint8)


def test_perturbations_shapes():
    out = apply_perturbations(_img(), CONFIG)
    assert set(out) == {"blur", "skew", "lowres"}
    for arr in out.values():
        assert arr.shape == (50, 80, 3)  # lowres restores original size
        assert arr.dtype == np.uint8


def _score(page, engine, cer, conf):
    return PageScore(
        page=page, engine=engine, script="Latin", has_ground_truth=True, scored=True,
        cer=cer, wer=cer, ref_free={"engine_confidence": conf},
    )


def test_silent_failure_flagged():
    clean = [_score("p1", EngineName.tesseract, cer=0.05, conf=0.90)]
    degraded = {"blur": [_score("p1", EngineName.tesseract, cer=0.45, conf=0.88)]}
    rows = compute_robustness(clean, degraded)
    assert len(rows) == 1
    r = rows[0]
    assert r.silent_failure is True          # CER jumped, confidence held
    assert r.delta_cer > 0.3
    assert "confident-but-wrong" in r.note


def test_honest_failure_not_flagged():
    clean = [_score("p1", EngineName.tesseract, cer=0.05, conf=0.90)]
    degraded = {"skew": [_score("p1", EngineName.tesseract, cer=0.45, conf=0.40)]}
    rows = compute_robustness(clean, degraded)
    assert rows[0].silent_failure is False   # confidence dropped with accuracy


def test_llm_no_confidence_is_undetectable():
    clean = [_score("p1", EngineName.vision_llm, cer=0.05, conf=None)]
    degraded = {"lowres": [_score("p1", EngineName.vision_llm, cer=0.45, conf=None)]}
    rows = compute_robustness(clean, degraded)
    assert rows[0].silent_failure is None    # honest: cannot assess via confidence
    assert "undetectable" in rows[0].note


if __name__ == "__main__":
    for fn in [
        test_perturbations_shapes,
        test_silent_failure_flagged,
        test_honest_failure_not_flagged,
        test_llm_no_confidence_is_undetectable,
    ]:
        fn()
        print(f"OK: {fn.__name__}")
    print("\nAll Step 7 robustness checks passed.")
