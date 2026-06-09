"""Step 2 unit check: every record type survives write -> read -> validate.

Runnable two ways:
    pytest tests/test_schemas_roundtrip.py
    python -m tests.test_schemas_roundtrip      (no pytest needed)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from cascade_eval.logger import CascadeLogger, read_jsonl
from cascade_eval.schemas import (
    AbsenceRecord,
    BBox,
    DiscardRecord,
    EngineName,
    EngineResult,
    RoutingDecision,
    TokenConfidence,
)


def _sample_records():
    engine = EngineResult(
        page="ara_001",
        engine=EngineName.tesseract,
        text="مرحبا",
        confidence=0.91,
        tokens=[TokenConfidence(text="مرحبا", confidence=0.91, bbox=BBox(x=10, y=20, w=80, h=30))],
        runtime_s=0.42,
    )
    routing = RoutingDecision(
        page="ara_001",
        script="Arabic",
        is_rtl=True,
        has_ground_truth=True,
        candidates_considered=[EngineName.tesseract, EngineName.easyocr],
        winner=EngineName.tesseract,
        winning_text="مرحبا",
        winning_confidence=0.91,
        proxy_score=0.78,
        thresholds={"tau_high": 0.85, "tau_min": 0.5, "delta_dict": 0.4},
        reason="tesseract conf 0.91 >= tau_high 0.85 and dict-hit 0.78 >= delta 0.4",
    )
    discard = DiscardRecord(
        page="ara_001",
        engine=EngineName.easyocr,
        confidence=0.64,
        proxy_score=0.55,
        discard_reason="lower confidence than tesseract winner",
    )
    absence = AbsenceRecord(
        page="cop_004",
        region=BBox(x=0, y=400, w=600, h=120),
        reason="all engines below tau_min 0.5 on low-resource Coptic region",
        best_effort_text="ⲡⲣⲱⲙⲉ?",
        best_effort_engine=EngineName.vision_llm,
    )
    return engine, routing, discard, absence


def test_roundtrip():
    engine, routing, discard, absence = _sample_records()
    with tempfile.TemporaryDirectory() as d:
        results = Path(d)
        with CascadeLogger(results) as log:
            log.engine_result(engine)
            log.routing(routing)
            log.discard(discard)
            log.absence(absence)

        assert read_jsonl(results / "engine_results.jsonl", EngineResult) == [engine]
        assert read_jsonl(results / "routing_log.jsonl", RoutingDecision) == [routing]
        assert read_jsonl(results / "discards.jsonl", DiscardRecord) == [discard]
        assert read_jsonl(results / "absences.jsonl", AbsenceRecord) == [absence]


if __name__ == "__main__":
    test_roundtrip()
    print("OK: all four record types round-tripped (write -> read -> validate)")
