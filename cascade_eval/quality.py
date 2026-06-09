"""Reference-free quality proxies.

These estimate how good a transcription is WITHOUT ground truth — the core of
the "measure cascade quality when there is no answer key" problem. None of
these is CER; each is explicitly an estimate, and each reports whether it could
even be computed (so the harness never silently fakes a number).

Three independent signals, chosen so coverage degrades gracefully:
  - engine_confidence : what the model thinks (None for the vision-LLM — no API
                        confidence; that's the confident-vs-correct gap).
  - dictionary_hit    : fraction of tokens in a known wordlist. Powerful for
                        well-resourced languages; None when no wordlist exists
                        (the low-resource case — flagged, not faked).
  - script_consistency: fraction of letters in the EXPECTED script. Needs no
                        external resource, so it still works for Coptic/Jawi
                        where no dictionary is available. Catches an engine that
                        emits fluent gibberish in the wrong script.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .schemas import EngineResult
from .script_detect import detect_script

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass
class QualityReport:
    """Reference-free assessment of one candidate. Every field is optional —
    None means 'could not be computed', distinct from 0.0 ('computed, bad')."""

    engine_confidence: float | None = None
    dictionary_hit: float | None = None
    script_consistency: float | None = None
    n_tokens: int = 0

    def combined(self) -> float | None:
        """Average of the signals that COULD be computed, or None if none could.
        Kept simple and legible on purpose — the router logs the components, not
        just this scalar, so a reader can see what went into it."""
        parts = [
            v for v in (self.engine_confidence, self.dictionary_hit, self.script_consistency)
            if v is not None
        ]
        return sum(parts) / len(parts) if parts else None

    def as_dict(self) -> dict:
        return {
            "engine_confidence": self.engine_confidence,
            "dictionary_hit": self.dictionary_hit,
            "script_consistency": self.script_consistency,
            "n_tokens": self.n_tokens,
            "combined": self.combined(),
        }


def load_wordlist(lang: str, wordlists_dir: str | Path) -> set[str] | None:
    """Load data/wordlists/<lang>.txt (one token per line). Returns None if the
    list doesn't exist — the honest 'no dictionary for this language' case."""
    path = Path(wordlists_dir) / f"{lang}.txt"
    if not path.exists():
        return None
    words = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        w = line.strip().lower()
        if w:
            words.add(w)
    return words or None


def dictionary_hit_rate(text: str, wordlist: set[str] | None) -> float | None:
    """Fraction of tokens present in the wordlist. None if no wordlist."""
    if wordlist is None:
        return None
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in wordlist)
    return hits / len(tokens)


def script_consistency(text: str, expected_script: str) -> float | None:
    """Fraction of script-bearing characters that are in the expected script.
    None if the text has no script-bearing characters at all."""
    profile = detect_script(text)
    if profile.script_char_total == 0:
        return None
    return profile.proportion(expected_script)


def assess(
    result: EngineResult,
    expected_script: str,
    wordlist: set[str] | None = None,
) -> QualityReport:
    """Compute all available reference-free proxies for one engine result."""
    text = result.text or ""
    return QualityReport(
        engine_confidence=result.confidence,  # None for vision-LLM, by design
        dictionary_hit=dictionary_hit_rate(text, wordlist),
        script_consistency=script_consistency(text, expected_script),
        n_tokens=len(_TOKEN_RE.findall(text)),
    )
