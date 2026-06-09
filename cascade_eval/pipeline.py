"""End-to-end corpus orchestration: preprocess -> route -> score -> robustness.

Decoupled from disk IO so it can be driven by a real manifest (run.py) or by
synthetic pages (tests). Produces everything the report layer needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .engines import Engine
from .logger import CascadeLogger, NullLogger
from .metrics import PageScore, aggregate, score_result
from .perturb import apply_perturbations
from .preprocess import preprocess
from .robustness import RobustnessRow, compute_robustness
from .quality import load_wordlist
from .router import Router
from .schemas import EngineName, RoutingDecision


@dataclass
class PageItem:
    page: str
    image: np.ndarray
    script: str
    language: str
    ground_truth: str | None = None
    is_handwritten: bool = False
    in_robustness_subset: bool = False


@dataclass
class CorpusResult:
    decisions: list[RoutingDecision] = field(default_factory=list)
    scores: list[PageScore] = field(default_factory=list)       # per engine, clean pages
    robustness_rows: list[RobustnessRow] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)


def _score_candidates(router: Router, item: PageItem, wordlist) -> list[PageScore]:
    """Score every engine that ran on this page against its ground truth."""
    out = []
    for c in router.last_candidates:
        out.append(
            score_result(
                c.result, script=item.script, ground_truth=item.ground_truth, ref_free=c.quality
            )
        )
    return out


def run_corpus(
    config: dict, engines: dict[EngineName, Engine], logger: CascadeLogger, pages: list[PageItem]
) -> CorpusResult:
    router = Router(engines, config, logger)
    result = CorpusResult()

    # Routing happens against a clean preprocess of each page; the robustness
    # subset is additionally re-run through degraded variants (on a NullLogger,
    # so the clean run's provenance logs stay clean).
    degraded_router = Router(engines, config, NullLogger())
    clean_subset_scores: list[PageScore] = []
    degraded_scores: dict[str, list[PageScore]] = {}

    for item in pages:
        wordlist = load_wordlist(item.language, config.get("paths", {}).get("wordlists", "data/wordlists"))
        pre = preprocess(item.image)
        decision = router.route(
            pre, item.page, script=item.script, language=item.language,
            has_ground_truth=item.ground_truth is not None, is_handwritten=item.is_handwritten,
        )
        result.decisions.append(decision)
        page_scores = _score_candidates(router, item, wordlist)
        result.scores.extend(page_scores)

        # Robustness: only meaningful where we can measure CER (GT present).
        if item.in_robustness_subset and item.ground_truth is not None:
            clean_subset_scores.extend(page_scores)
            for name, degraded_img in apply_perturbations(item.image, config).items():
                degraded_pre = preprocess(degraded_img)
                degraded_router.route(
                    degraded_pre, item.page, script=item.script, language=item.language,
                    has_ground_truth=True, is_handwritten=item.is_handwritten,
                )
                degraded_scores.setdefault(name, []).extend(
                    _score_candidates(degraded_router, item, wordlist)
                )

    result.robustness_rows = compute_robustness(clean_subset_scores, degraded_scores)
    result.aggregate = aggregate(result.scores)
    return result
