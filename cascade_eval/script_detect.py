"""Script detection by Unicode-block analysis.

Deliberately NOT a learned language-ID model. The router needs to explain every
decision, so script detection is a transparent histogram: count each character
into its Unicode script block and report the distribution. You can read exactly
why a page was called "Arabic" — N% of its letters live in the Arabic blocks.

Two uses in the cascade:
  1. Confirm/derive a page's script for engine+language selection.
  2. As a routing signal: if an engine's OUTPUT is mostly the wrong script
     (e.g. Tesseract emits Latin gibberish on an Arabic page), that script
     mismatch is a legible reason to distrust/discard the candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# (script_name, is_rtl, [(lo, hi), ...]) — inclusive codepoint ranges.
_SCRIPT_RANGES: list[tuple[str, bool, list[tuple[int, int]]]] = [
    ("Latin", False, [(0x41, 0x5A), (0x61, 0x7A), (0xC0, 0x24F), (0x1E00, 0x1EFF)]),
    ("Arabic", True, [(0x600, 0x6FF), (0x750, 0x77F), (0x8A0, 0x8FF),
                      (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]),
    ("Coptic", False, [(0x2C80, 0x2CFF), (0x3E2, 0x3EF)]),
    ("Greek", False, [(0x370, 0x3E1), (0x3F0, 0x3FF)]),  # exclude Coptic-in-Greek block
    ("Cyrillic", False, [(0x400, 0x4FF)]),
    ("Hebrew", True, [(0x590, 0x5FF)]),
    ("Han", False, [(0x4E00, 0x9FFF)]),
]

_RTL_SCRIPTS = {name for name, rtl, _ in _SCRIPT_RANGES if rtl}

# Some language labels are written in another script's Unicode block. Jawi
# (Malay) uses the Arabic block, so when measuring script-consistency for a
# page labeled "Jawi" we compare against the Arabic block, while keeping "Jawi"
# as the reporting/grouping label so it stays distinct from Arabic in metrics.
_BLOCK_ALIAS = {"Jawi": "Arabic"}


def resolve_block(label: str) -> str:
    """Map a language/script label to the Unicode-block name used to measure it."""
    return _BLOCK_ALIAS.get(label, label)


def char_script(ch: str) -> str | None:
    """Return the script name for a single character, or None for
    digits/punctuation/whitespace/symbols (not counted toward any script)."""
    cp = ord(ch)
    for name, _rtl, ranges in _SCRIPT_RANGES:
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return name
    return None


@dataclass
class ScriptProfile:
    dominant: str | None  # most common script, or None if no script chars
    is_rtl: bool
    histogram: dict[str, int] = field(default_factory=dict)  # script -> char count
    script_char_total: int = 0  # total chars that belong to *some* script

    def proportion(self, script: str) -> float:
        if self.script_char_total == 0:
            return 0.0
        return self.histogram.get(script, 0) / self.script_char_total

    @property
    def dominant_proportion(self) -> float:
        return self.proportion(self.dominant) if self.dominant else 0.0


def detect_script(text: str) -> ScriptProfile:
    """Build a Unicode-block histogram of `text` and report the dominant script."""
    hist: dict[str, int] = {}
    for ch in text:
        s = char_script(ch)
        if s is not None:
            hist[s] = hist.get(s, 0) + 1
    total = sum(hist.values())
    dominant = max(hist, key=hist.get) if hist else None
    return ScriptProfile(
        dominant=dominant,
        is_rtl=dominant in _RTL_SCRIPTS if dominant else False,
        histogram=hist,
        script_char_total=total,
    )


def is_rtl(script: str) -> bool:
    return resolve_block(script) in _RTL_SCRIPTS
