"""CER / WER scoring — and honest handling of pages with no ground truth.

The non-negotiable rule of this module: a CER/WER number is produced ONLY when
real reference text exists. For a page with no ground truth we return cer=wer=
None and attach the reference-free estimate instead, explicitly flagged. We
never fabricate a score for an ungrounded page.

CER/WER are computed with `jiwer` when installed (the production path); a small
internal edit-distance fallback keeps the harness runnable and testable without
it. Both apply the same documented normalization so the two paths agree.

Scoring is done PER ENGINE (against the same ground truth), which is what makes
the per-engine / per-script comparison in the report meaningful — not just the
winner's score.
"""

from __future__ import annotations

import importlib.util
import unicodedata
from dataclasses import dataclass, field

from .quality import QualityReport
from .schemas import EngineName, EngineResult

_HAVE_JIWER = importlib.util.find_spec("jiwer") is not None


def normalize(text: str) -> str:
    """Documented normalization applied to BOTH reference and hypothesis before
    scoring: Unicode NFC, collapse all runs of whitespace to single spaces,
    strip ends. Case and diacritics are preserved — for OCR/HTR a wrong case or
    missing diacritic is a real error, not noise to be normalized away."""
    text = unicodedata.normalize("NFC", text or "")
    return " ".join(text.split())


def _levenshtein(ref: list, hyp: list) -> int:
    """Edit distance between two sequences (chars or words)."""
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cost = 0 if r == h else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate over normalized text. Edit distance / len(reference)."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    if _HAVE_JIWER:
        import jiwer

        return float(jiwer.cer(ref, hyp))
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(list(ref), list(hyp)) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate over normalized text. Word edit distance / len(ref words)."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    if _HAVE_JIWER:
        import jiwer

        return float(jiwer.wer(ref, hyp))
    rw, hw = ref.split(), hyp.split()
    if not rw:
        return 0.0 if not hw else 1.0
    return _levenshtein(rw, hw) / len(rw)


@dataclass
class PageScore:
    """One engine's result on one page. `scored` distinguishes a real CER/WER
    from a flagged reference-free-only estimate."""

    page: str
    engine: EngineName
    script: str
    has_ground_truth: bool
    scored: bool
    cer: float | None = None
    wer: float | None = None
    ref_free: dict | None = None  # reference-free proxies (always available)
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "page": self.page,
            "engine": self.engine.value,
            "script": self.script,
            "has_ground_truth": self.has_ground_truth,
            "scored": self.scored,
            "cer": self.cer,
            "wer": self.wer,
            "ref_free": self.ref_free,
            "note": self.note,
        }


def score_result(
    result: EngineResult,
    *,
    script: str,
    ground_truth: str | None,
    ref_free: QualityReport | None = None,
) -> PageScore:
    """Score one engine result. Computes CER/WER only when ground truth exists;
    otherwise returns an explicitly-flagged reference-free-only estimate."""
    rf = ref_free.as_dict() if ref_free is not None else None
    if ground_truth is None:
        return PageScore(
            page=result.page, engine=result.engine, script=script,
            has_ground_truth=False, scored=False, cer=None, wer=None, ref_free=rf,
            note="NO GROUND TRUTH — reference-free estimate only; CER/WER not computed",
        )
    return PageScore(
        page=result.page, engine=result.engine, script=script,
        has_ground_truth=True, scored=True,
        cer=cer(ground_truth, result.text), wer=wer(ground_truth, result.text),
        ref_free=rf, note="scored against ground truth",
    )


@dataclass
class Aggregate:
    n_pages: int = 0
    n_scored: int = 0          # had ground truth -> contributed to CER/WER
    n_unscored: int = 0        # no ground truth -> reference-free only
    mean_cer: float | None = None
    mean_wer: float | None = None

    def as_dict(self) -> dict:
        return {
            "n_pages": self.n_pages, "n_scored": self.n_scored,
            "n_unscored": self.n_unscored, "mean_cer": self.mean_cer,
            "mean_wer": self.mean_wer,
        }


def _aggregate(scores: list[PageScore]) -> Aggregate:
    agg = Aggregate(n_pages=len(scores))
    scored = [s for s in scores if s.scored]
    agg.n_scored = len(scored)
    agg.n_unscored = agg.n_pages - agg.n_scored
    if scored:
        agg.mean_cer = sum(s.cer for s in scored) / len(scored)
        agg.mean_wer = sum(s.wer for s in scored) / len(scored)
    # mean_cer/mean_wer stay None when nothing was scored — honest, not 0.0
    return agg


def aggregate(scores: list[PageScore]) -> dict:
    """Roll page scores up per-engine, per-script, and per (engine, script).
    Means are computed only over scored pages; unscored pages are counted, not
    folded into a fake average."""
    by_engine: dict[str, list[PageScore]] = {}
    by_script: dict[str, list[PageScore]] = {}
    by_engine_script: dict[str, list[PageScore]] = {}
    for s in scores:
        by_engine.setdefault(s.engine.value, []).append(s)
        by_script.setdefault(s.script, []).append(s)
        by_engine_script.setdefault(f"{s.engine.value}|{s.script}", []).append(s)
    return {
        "per_engine": {k: _aggregate(v).as_dict() for k, v in by_engine.items()},
        "per_script": {k: _aggregate(v).as_dict() for k, v in by_script.items()},
        "per_engine_script": {k: _aggregate(v).as_dict() for k, v in by_engine_script.items()},
    }
