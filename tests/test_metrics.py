"""Step 6: CER/WER on known pairs, plus the no-ground-truth honesty path.

Run:
    python -m tests.test_metrics
    pytest tests/test_metrics.py
"""

from __future__ import annotations

from cascade_eval.metrics import aggregate, cer, score_result, wer
from cascade_eval.quality import QualityReport
from cascade_eval.schemas import EngineName, EngineResult


def _approx(a, b, eps=1e-9):
    return abs(a - b) < eps


def test_cer_known_pairs():
    assert cer("hello", "hello") == 0.0
    assert _approx(cer("hello", "hallo"), 1 / 5)       # 1 substitution / 5 chars
    assert _approx(cer("abc", "ab"), 1 / 3)            # 1 deletion / 3 chars


def test_wer_known_pairs():
    assert wer("the cat sat", "the cat sat") == 0.0
    assert _approx(wer("the cat sat", "the dog sat"), 1 / 3)   # 1 of 3 words
    assert _approx(wer("a b c d", "a b c"), 1 / 4)


def test_normalization_collapses_whitespace():
    # Extra whitespace and NFC differences must not count as errors.
    assert cer("hello world", "hello   world\n") == 0.0


def test_scored_when_ground_truth_present():
    res = EngineResult(page="lat_001", engine=EngineName.tesseract, text="hallo", confidence=0.9)
    score = score_result(res, script="Latin", ground_truth="hello")
    assert score.scored is True
    assert score.has_ground_truth is True
    assert _approx(score.cer, 1 / 5)
    assert score.note == "scored against ground truth"


def test_no_ground_truth_is_flagged_not_faked():
    res = EngineResult(page="cop_001", engine=EngineName.vision_llm, text="ⲡⲣⲱⲙⲉ", confidence=None)
    rf = QualityReport(engine_confidence=None, dictionary_hit=None, script_consistency=1.0, n_tokens=1)
    score = score_result(res, script="Coptic", ground_truth=None, ref_free=rf)
    assert score.scored is False
    assert score.cer is None and score.wer is None       # NOT a fabricated number
    assert "NO GROUND TRUTH" in score.note
    assert score.ref_free["script_consistency"] == 1.0   # estimate still surfaced


def test_aggregate_separates_scored_and_unscored():
    scores = [
        score_result(EngineResult(page="lat_001", engine=EngineName.tesseract, text="hello"),
                     script="Latin", ground_truth="hello"),                       # cer 0
        score_result(EngineResult(page="lat_002", engine=EngineName.tesseract, text="wrld"),
                     script="Latin", ground_truth="world"),                       # cer 0.2
        score_result(EngineResult(page="cop_001", engine=EngineName.tesseract, text="x"),
                     script="Coptic", ground_truth=None),                         # unscored
    ]
    agg = aggregate(scores)
    eng = agg["per_engine"]["tesseract"]
    assert eng["n_pages"] == 3
    assert eng["n_scored"] == 2
    assert eng["n_unscored"] == 1
    assert _approx(eng["mean_cer"], (0.0 + 0.2) / 2)     # mean over scored only
    # The Coptic (no-GT) page contributes a count but no fabricated CER.
    assert agg["per_script"]["Coptic"]["mean_cer"] is None
    assert agg["per_script"]["Coptic"]["n_unscored"] == 1


if __name__ == "__main__":
    for fn in [
        test_cer_known_pairs,
        test_wer_known_pairs,
        test_normalization_collapses_whitespace,
        test_scored_when_ground_truth_present,
        test_no_ground_truth_is_flagged_not_faked,
        test_aggregate_separates_scored_and_unscored,
    ]:
        fn()
        print(f"OK: {fn.__name__}")
    print("\nAll Step 6 metric checks passed.")
