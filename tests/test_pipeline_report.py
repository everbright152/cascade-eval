"""Step 8: full pipeline -> report. Drives preprocess->route->score->robustness
with fake engines and synthetic pages, then writes all artifacts and checks
they exist and contain the right shape. Proves "one run produces all tables".

Run:
    python -m tests.test_pipeline_report
    pytest tests/test_pipeline_report.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from cascade_eval.logger import CascadeLogger
from cascade_eval.pipeline import PageItem, run_corpus
from cascade_eval.report import print_summary, write_all
from cascade_eval.schemas import EngineName, EngineResult

CONFIG = {
    "routing": {"tau_high": 0.85, "tau_min": 0.50, "delta_dict": 0.40,
                "low_resource_scripts": ["Coptic", "Arabic"]},
    "paths": {"wordlists": "data/wordlists"},
    "robustness": {"perturbations": {"blur": {"kernel": 5}, "skew": {"degrees": 7},
                                     "lowres": {"scale": 0.5}}},
}
COPTIC = "ⲡⲣⲱⲙⲉ ⲛⲟⲩⲧⲉ"


class FakeEngine:
    def __init__(self, name, outputs, default=("", None), supports=True):
        self.name = name
        self._outputs = outputs
        self._default = default
        self._supports = supports

    @property
    def is_available(self):
        return True

    def for_script(self, script):
        return self if self._supports else None

    def recognize(self, image, page):
        text, conf = self._outputs.get(page, self._default)
        return EngineResult(page=page, engine=self.name, text=text, confidence=conf, available=True)


def _img():
    return np.random.default_rng(2).integers(0, 256, (40, 80, 3)).astype(np.uint8)


def _run():
    engines = {
        EngineName.tesseract: FakeEngine(EngineName.tesseract,
            {"lat_001": ("hello world", 0.95), "cop_001": ("asdf qwer", 0.20)}),
        EngineName.easyocr: FakeEngine(EngineName.easyocr,
            {"lat_001": ("hello world", 0.90), "cop_001": ("", 0.0)}),
        EngineName.vision_llm: FakeEngine(EngineName.vision_llm,
            {"cop_001": (COPTIC, None)}),
    }
    pages = [
        PageItem("lat_001", _img(), "Latin", "eng", ground_truth="hello world",
                 in_robustness_subset=True),
        PageItem("cop_001", _img(), "Coptic", "cop", ground_truth=None),  # thin GT
    ]
    d = Path(tempfile.mkdtemp())
    log = CascadeLogger(d)
    result = run_corpus(CONFIG, engines, log, pages)
    log.close()
    return result, d


def test_pipeline_produces_results_and_artifacts():
    result, d = _run()

    # Two pages routed; Coptic resolved by the LLM (no absence here).
    assert len(result.decisions) == 2
    cop = [x for x in result.decisions if x.page == "cop_001"][0]
    assert cop.winner == EngineName.vision_llm
    assert cop.has_ground_truth is False

    # Latin scored (CER 0); Coptic unscored (no GT) — not faked.
    lat_t = [s for s in result.scores if s.page == "lat_001" and s.engine == EngineName.tesseract][0]
    assert lat_t.scored is True and lat_t.cer == 0.0
    assert all(not s.scored for s in result.scores if s.page == "cop_001")

    # Robustness: 3 perturbations on the one GT page.
    assert {r.perturbation for r in result.robustness_rows} == {"blur", "skew", "lowres"}

    # Artifacts written.
    written = write_all(result, d)
    for fname in ("per_engine_metrics.csv", "per_script_metrics.csv",
                  "robustness_slice.csv", "metrics.md"):
        assert (d / fname).exists() and (d / fname).stat().st_size >= 0
    md = (d / "metrics.md").read_text()
    assert "per engine" in md and "Robustness slice" in md
    print_summary(result, written)  # must not raise


if __name__ == "__main__":
    test_pipeline_produces_results_and_artifacts()
    print("\nOK: pipeline + report end-to-end")
