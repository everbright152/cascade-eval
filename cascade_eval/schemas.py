"""Output-contract schemas for the cascade.

These pydantic models ARE the deliverable's provenance layer. Every routing
decision, every discarded candidate, and every unrecoverable region is one of
these records, serialized to JSONL. Locking the shapes here — before any
routing logic exists — keeps discards and documented absence first-class
outputs rather than debugging afterthoughts.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class EngineName(str, Enum):
    tesseract = "tesseract"
    easyocr = "easyocr"
    vision_llm = "vision_llm"


class BBox(BaseModel):
    """Pixel bounding box, origin top-left."""

    x: int
    y: int
    w: int
    h: int


class TokenConfidence(BaseModel):
    """Per-word/box confidence from an engine — the raw signal the router and
    the confidence-vs-correctness analysis are built on."""

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BBox | None = None


class EngineResult(BaseModel):
    """One engine's attempt at one page. Logged for EVERY engine run, including
    engines that were unavailable (e.g. LLM fallback with no API key) — absence
    of a candidate is itself recorded."""

    page: str  # image id, e.g. "ara_001"
    engine: EngineName
    text: str = ""
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)  # page-level aggregate
    tokens: list[TokenConfidence] = Field(default_factory=list)
    runtime_s: float = 0.0
    available: bool = True  # False -> engine skipped; see `note`
    note: str | None = None  # why unavailable, or any caveat


class RoutingDecision(BaseModel):
    """The legible routing record: who won, why, and under which thresholds.
    One per page. `winner is None` means the page was ruled unrecoverable and a
    matching AbsenceRecord was emitted."""

    page: str
    script: str
    is_rtl: bool = False
    has_ground_truth: bool = False
    candidates_considered: list[EngineName] = Field(default_factory=list)
    winner: EngineName | None = None
    winning_text: str | None = None
    winning_confidence: float | None = None
    proxy_score: float | None = None  # reference-free quality of the winner
    thresholds: dict = Field(default_factory=dict)  # tau_high/tau_min/delta in effect
    reason: str  # human-readable: why this winner under these thresholds


class DiscardRecord(BaseModel):
    """A candidate the router rejected. First-class output: one per discarded
    engine result, with the explicit reason it lost."""

    page: str
    engine: EngineName
    confidence: float | None = None
    proxy_score: float | None = None
    discard_reason: str


class AbsenceRecord(BaseModel):
    """Documented absence: a page or region the cascade could not recover with
    confidence. Surfaced so a downstream reader knows precisely what is missing
    and why — never silently dropped."""

    page: str
    region: BBox | None = None  # None = whole page unrecoverable
    reason: str
    best_effort_text: str | None = None  # best guess, explicitly low-confidence
    best_effort_engine: EngineName | None = None
