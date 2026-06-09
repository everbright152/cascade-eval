"""The cascade router — the core, legible decision layer.

Policy (mirrors the design doc's routing diagram), driven entirely by the
explicit thresholds in config.yaml so the logic stays auditable:

  1. Tesseract first. Accept outright if confidence >= tau_high AND the lexical
     signal (dictionary-hit, or script-consistency when no wordlist) >= delta.
  2. Else EasyOCR (skipped + logged when it has no model for the script).
     Accept outright on the same tau_high + lexical gate.
  3. Else, for low-resource or handwritten pages, escalate to the vision-LLM.
  4. Else pick the best-of-available by combined reference-free proxy, but only
     if it clears tau_min.
  5. Else the page is ruled UNRECOVERABLE -> AbsenceRecord (documented absence).

Provenance is not optional here: every engine run is logged (including engines
that couldn't run), every non-winning candidate becomes a DiscardRecord with a
human-readable reason, and every decision — win or absence — becomes a
RoutingDecision recording the thresholds in force.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .engines import Engine
from .logger import CascadeLogger
from .quality import QualityReport, assess, load_wordlist
from .schemas import (
    AbsenceRecord,
    DiscardRecord,
    EngineName,
    EngineResult,
    RoutingDecision,
)
from .script_detect import is_rtl


@dataclass
class Candidate:
    engine: EngineName
    result: EngineResult
    quality: QualityReport

    @property
    def combined(self) -> float | None:
        return self.quality.combined()


def _lexical(q: QualityReport) -> float | None:
    """The lexical confirmation signal: dictionary-hit if we have a wordlist,
    else script-consistency (works for low-resource langs with no dictionary)."""
    return q.dictionary_hit if q.dictionary_hit is not None else q.script_consistency


class Router:
    def __init__(self, engines: dict[EngineName, Engine], config: dict, logger: CascadeLogger):
        self.engines = engines
        self.logger = logger
        r = config.get("routing", {})
        self.tau_high = r.get("tau_high", 0.85)
        self.tau_min = r.get("tau_min", 0.50)
        self.delta = r.get("delta_dict", 0.40)
        self.low_resource = set(r.get("low_resource_scripts", []))
        self.wordlists_dir = config.get("paths", {}).get("wordlists", "data/wordlists")
        # Candidates from the most recent route() call — lets the pipeline score
        # every engine without re-running them.
        self.last_candidates: list[Candidate] = []

    # --- helpers ------------------------------------------------------------

    def _accept_high(self, c: Candidate) -> bool:
        """Outright-accept gate: high confidence AND lexical confirmation."""
        conf = c.result.confidence
        lex = _lexical(c.quality)
        return conf is not None and conf >= self.tau_high and lex is not None and lex >= self.delta

    # --- main ---------------------------------------------------------------

    def route(
        self,
        image,
        page: str,
        *,
        script: str,
        language: str,
        has_ground_truth: bool = False,
        is_handwritten: bool = False,
    ) -> RoutingDecision:
        wordlist = load_wordlist(language, self.wordlists_dir)
        candidates: list[Candidate] = []

        def consider(name: EngineName) -> Candidate | None:
            engine = self.engines.get(name)
            if engine is None:
                return None
            eng = engine
            if hasattr(engine, "for_script"):
                eng = engine.for_script(script)
                if eng is None:  # e.g. EasyOCR has no Coptic model
                    self.logger.engine_result(
                        EngineResult(
                            page=page, engine=name, available=False,
                            note=f"no model for script '{script}'",
                        )
                    )
                    return None
            result = eng.recognize(image, page)
            self.logger.engine_result(result)
            if not result.available:
                return None  # dep missing / no key — logged, not a candidate
            cand = Candidate(name, result, assess(result, script, wordlist))
            candidates.append(cand)
            return cand

        winner: Candidate | None = None
        reason = ""

        # Stage 1 — Tesseract baseline.
        c_t = consider(EngineName.tesseract)
        if c_t and self._accept_high(c_t):
            winner = c_t
            reason = (
                f"tesseract cleared tau_high (conf={c_t.result.confidence:.2f}>={self.tau_high}, "
                f"lexical={_lexical(c_t.quality):.2f}>={self.delta})"
            )

        # Stage 2 — EasyOCR contrast.
        if winner is None:
            c_e = consider(EngineName.easyocr)
            if c_e and self._accept_high(c_e):
                winner = c_e
                reason = (
                    f"easyocr cleared tau_high (conf={c_e.result.confidence:.2f}>={self.tau_high}, "
                    f"lexical={_lexical(c_e.quality):.2f}>={self.delta}); tesseract did not"
                )

        # Stage 3 — vision-LLM fallback for low-resource / handwritten pages.
        if winner is None:
            eligible = script in self.low_resource or is_handwritten
            if eligible:
                consider(EngineName.vision_llm)

        # Stage 4 — best-of-available, gated by tau_min.
        if winner is None:
            scored = [c for c in candidates if c.combined is not None]
            if scored:
                best = max(scored, key=lambda c: c.combined)
                if best.combined >= self.tau_min:
                    winner = best
                    reason = (
                        f"no engine cleared tau_high; selected best-of-available "
                        f"{best.engine.value} by combined proxy "
                        f"({best.combined:.2f}>=tau_min {self.tau_min})"
                    )
                else:
                    reason = (
                        f"all {len(scored)} candidates below tau_min "
                        f"(best={best.engine.value} combined={best.combined:.2f}<{self.tau_min})"
                    )
            else:
                reason = "no candidate produced a computable quality proxy"

        self.last_candidates = candidates
        return self._finalize(candidates, winner, page, script, has_ground_truth, reason)

    # --- finalize: write all provenance records -----------------------------

    def _finalize(
        self,
        candidates: list[Candidate],
        winner: Candidate | None,
        page: str,
        script: str,
        has_ground_truth: bool,
        reason: str,
    ) -> RoutingDecision:
        thresholds = {"tau_high": self.tau_high, "tau_min": self.tau_min, "delta_dict": self.delta}

        # Discards: every candidate that isn't the winner, with a reason.
        for c in candidates:
            if c is winner:
                continue
            if winner is None:
                d_reason = f"page unrecoverable: {reason}"
            else:
                w = winner.combined
                d_reason = (
                    f"not selected: combined={c.combined if c.combined is not None else 'n/a'}, "
                    f"conf={c.result.confidence}; winner {winner.engine.value} "
                    f"combined={w:.2f}" if w is not None else
                    f"not selected: winner {winner.engine.value}"
                )
            self.logger.discard(
                DiscardRecord(
                    page=page, engine=c.engine,
                    confidence=c.result.confidence, proxy_score=c.combined,
                    discard_reason=d_reason,
                )
            )

        if winner is None:
            # Documented absence: keep the best-effort guess but flag it.
            best_effort = max(candidates, key=lambda c: (c.combined or -1.0), default=None)
            self.logger.absence(
                AbsenceRecord(
                    page=page, region=None, reason=reason,
                    best_effort_text=best_effort.result.text if best_effort else None,
                    best_effort_engine=best_effort.engine if best_effort else None,
                )
            )
            decision = RoutingDecision(
                page=page, script=script, is_rtl=is_rtl(script),
                has_ground_truth=has_ground_truth,
                candidates_considered=[c.engine for c in candidates],
                winner=None, winning_text=None, winning_confidence=None, proxy_score=None,
                thresholds=thresholds, reason=reason,
            )
        else:
            decision = RoutingDecision(
                page=page, script=script, is_rtl=is_rtl(script),
                has_ground_truth=has_ground_truth,
                candidates_considered=[c.engine for c in candidates],
                winner=winner.engine, winning_text=winner.result.text,
                winning_confidence=winner.result.confidence, proxy_score=winner.combined,
                thresholds=thresholds, reason=reason,
            )

        self.logger.routing(decision)
        return decision
