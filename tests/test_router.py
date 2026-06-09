"""Step 5: force every routing branch with fake engines and assert the logs.

Fake engines return canned (text, confidence) so we can drive the policy
deterministically and verify that routing decisions, discards, and absences are
all written as first-class output.

Run:
    python -m tests.test_router
    pytest tests/test_router.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from cascade_eval.logger import CascadeLogger, read_jsonl
from cascade_eval.router import Router
from cascade_eval.schemas import (
    AbsenceRecord,
    DiscardRecord,
    EngineName,
    EngineResult,
    RoutingDecision,
)

ARABIC = "مرحبا بالعالم"
COPTIC = "ⲡⲣⲱⲙⲉ ⲛⲟⲩⲧⲉ"
LATIN_GIBBERISH = "asdf qwer zxcv"

CONFIG = {
    "routing": {
        "tau_high": 0.85, "tau_min": 0.50, "delta_dict": 0.40,
        "low_resource_scripts": ["Coptic", "Jawi", "Arabic"],
    },
    "paths": {"wordlists": "data/wordlists"},  # no wordlists present -> uses script-consistency
}


class FakeEngine:
    """Minimal Engine stand-in driven by canned outputs."""

    def __init__(self, name: EngineName, text: str, confidence, supports: bool = True):
        self.name = name
        self._text = text
        self._conf = confidence
        self._supports = supports

    @property
    def is_available(self) -> bool:
        return True

    def for_script(self, script: str):
        return self if self._supports else None

    def recognize(self, image, page: str) -> EngineResult:
        return EngineResult(
            page=page, engine=self.name, text=self._text,
            confidence=self._conf, available=True,
        )


def _route(engines, **kw):
    """Run one page through a fresh logger; return (decision, results dir)."""
    d = tempfile.mkdtemp()
    log = CascadeLogger(Path(d))
    router = Router(engines, CONFIG, log)
    decision = router.route(None, kw.pop("page", "p_001"), **kw)
    log.close()
    return decision, Path(d)


def test_tesseract_wins_high_confidence():
    engines = {
        EngineName.tesseract: FakeEngine(EngineName.tesseract, ARABIC, 0.95),
        EngineName.easyocr: FakeEngine(EngineName.easyocr, ARABIC, 0.99),
    }
    decision, d = _route(engines, script="Arabic", language="ara")
    assert decision.winner == EngineName.tesseract
    # Cascade short-circuits: easyocr never runs -> only one engine result, no discards.
    assert read_jsonl(d / "discards.jsonl", DiscardRecord) == []
    assert len(read_jsonl(d / "engine_results.jsonl", EngineResult)) == 1
    assert decision.is_rtl is True


def test_easyocr_wins_when_tesseract_low():
    engines = {
        EngineName.tesseract: FakeEngine(EngineName.tesseract, ARABIC, 0.50),  # below tau_high
        EngineName.easyocr: FakeEngine(EngineName.easyocr, ARABIC, 0.92),
    }
    decision, d = _route(engines, script="Arabic", language="ara")
    assert decision.winner == EngineName.easyocr
    discards = read_jsonl(d / "discards.jsonl", DiscardRecord)
    assert [r.engine for r in discards] == [EngineName.tesseract]
    assert discards[0].discard_reason  # has a stated reason


def test_llm_fallback_on_low_resource():
    engines = {
        EngineName.tesseract: FakeEngine(EngineName.tesseract, LATIN_GIBBERISH, 0.60),  # wrong script
        EngineName.easyocr: FakeEngine(EngineName.easyocr, "", 0.0, supports=False),     # no Coptic model
        EngineName.vision_llm: FakeEngine(EngineName.vision_llm, COPTIC, None),          # no confidence
    }
    decision, d = _route(engines, script="Coptic", language="cop")
    assert decision.winner == EngineName.vision_llm  # wins on script-consistency proxy
    # EasyOCR logged as unavailable (no model), not a candidate.
    eng_results = read_jsonl(d / "engine_results.jsonl", EngineResult)
    easy = [r for r in eng_results if r.engine == EngineName.easyocr][0]
    assert easy.available is False and "no model" in easy.note
    # Tesseract (wrong-script) discarded.
    discards = read_jsonl(d / "discards.jsonl", DiscardRecord)
    assert EngineName.tesseract in [r.engine for r in discards]


def test_unrecoverable_emits_absence():
    engines = {
        EngineName.tesseract: FakeEngine(EngineName.tesseract, LATIN_GIBBERISH, 0.30),
        EngineName.easyocr: FakeEngine(EngineName.easyocr, LATIN_GIBBERISH, 0.30),
    }
    # Arabic page, both engines emit low-confidence wrong-script text -> nothing clears tau_min.
    decision, d = _route(engines, script="Arabic", language="ara")
    assert decision.winner is None
    absences = read_jsonl(d / "absences.jsonl", AbsenceRecord)
    assert len(absences) == 1
    assert absences[0].region is None  # whole page
    assert absences[0].best_effort_text  # best-effort guess preserved, but flagged
    # Both candidates discarded with an "unrecoverable" reason.
    discards = read_jsonl(d / "discards.jsonl", DiscardRecord)
    assert len(discards) == 2
    assert all("unrecoverable" in r.discard_reason for r in discards)


if __name__ == "__main__":
    for fn in [
        test_tesseract_wins_high_confidence,
        test_easyocr_wins_when_tesseract_low,
        test_llm_fallback_on_low_resource,
        test_unrecoverable_emits_absence,
    ]:
        fn()
        print(f"OK: {fn.__name__}")
    print("\nAll Step 5 router branches exercised; logs verified.")
