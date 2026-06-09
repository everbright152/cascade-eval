"""Robustness slice: how the metrics move under degradation, and where the
cascade fails *silently*.

The headline signal this module exists to surface: a perturbation that makes
CER climb while the engine's confidence stays high. That is the model being
confidently wrong — the exact failure the harness is meant to catch, because in
production (no ground truth) a high confidence is all you'd see.

For the vision-LLM, which reports no confidence at all, silent-failure-by-
confidence is *undetectable by construction* — we record that honestly (flag =
None) rather than pretend, and note that only a reference-free proxy could catch
it there.
"""

from __future__ import annotations

from dataclasses import dataclass

from .metrics import PageScore


def _conf(score: PageScore) -> float | None:
    if score.ref_free is None:
        return None
    return score.ref_free.get("engine_confidence")


@dataclass
class RobustnessRow:
    page: str
    engine: str
    perturbation: str
    clean_cer: float | None
    degraded_cer: float | None
    delta_cer: float | None
    clean_conf: float | None
    degraded_conf: float | None
    delta_conf: float | None
    silent_failure: bool | None  # None = cannot assess (no confidence signal)
    note: str = ""

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _flag_silent_failure(
    clean: PageScore, degraded: PageScore, cer_jump: float, conf_tol: float
) -> tuple[bool | None, str]:
    """Silent failure := CER rose materially while confidence did NOT drop."""
    if clean.cer is None or degraded.cer is None:
        return None, "not scored (no ground truth) — use reference-free proxy instead"
    cc, dc = _conf(clean), _conf(degraded)
    if dc is None or cc is None:
        return None, (
            "no confidence signal for this run — undetectable via confidence "
            "(always true for the vision-LLM; also when an engine returns no confident tokens)"
        )
    delta_cer = degraded.cer - clean.cer
    delta_conf = dc - cc
    if delta_cer >= cer_jump and delta_conf >= -conf_tol:
        return True, (
            f"CER rose {delta_cer:+.2f} but confidence held ({delta_conf:+.2f}) "
            "— confident-but-wrong"
        )
    return False, "confidence tracks accuracy" if delta_cer >= cer_jump else "robust"


def compute_robustness(
    clean_scores: list[PageScore],
    degraded_scores: dict[str, list[PageScore]],
    *,
    cer_jump: float = 0.10,
    conf_tol: float = 0.05,
) -> list[RobustnessRow]:
    """Build one row per (page, engine, perturbation) with clean-vs-degraded
    deltas and the silent-failure flag.

    clean_scores: PageScores on the clean images.
    degraded_scores: {perturbation_name: PageScores on that degraded variant}.
    """
    clean_idx = {(s.page, s.engine): s for s in clean_scores}
    rows: list[RobustnessRow] = []
    for perturbation, scores in degraded_scores.items():
        for d in scores:
            c = clean_idx.get((d.page, d.engine))
            if c is None:
                continue
            cc, dc = _conf(c), _conf(d)
            flag, note = _flag_silent_failure(c, d, cer_jump, conf_tol)
            rows.append(
                RobustnessRow(
                    page=d.page, engine=d.engine.value, perturbation=perturbation,
                    clean_cer=c.cer, degraded_cer=d.cer,
                    delta_cer=(d.cer - c.cer) if (c.cer is not None and d.cer is not None) else None,
                    clean_conf=cc, degraded_conf=dc,
                    delta_conf=(dc - cc) if (cc is not None and dc is not None) else None,
                    silent_failure=flag, note=note,
                )
            )
    return rows
