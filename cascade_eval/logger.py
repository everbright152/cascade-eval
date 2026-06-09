"""JSONL writers for the cascade's provenance records.

One file per record type, append-mode within a run, truncated at run start so
each run produces a clean, regenerable set of logs. Keeping these as plain
JSONL (not a DB) means they're greppable, diffable, and trivially inspected by
a reviewer — which is the point.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from .schemas import (
    AbsenceRecord,
    DiscardRecord,
    EngineResult,
    RoutingDecision,
)


class JsonlWriter:
    """Append pydantic records to a JSONL file as one compact object per line."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate on open so a run starts clean; subsequent writes append.
        self._fh = self.path.open("w", encoding="utf-8")

    def write(self, record: BaseModel) -> None:
        self._fh.write(record.model_dump_json() + "\n")
        self._fh.flush()  # flush so logs are durable even if a run crashes

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


def read_jsonl(path: Path, model: type[BaseModel]) -> list[BaseModel]:
    """Read a JSONL file back into validated pydantic records."""
    records: list[BaseModel] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(model.model_validate_json(line))
    return records


class CascadeLogger:
    """Bundles the four provenance streams the harness emits each run.

    Usage:
        with CascadeLogger(Path("results")) as log:
            log.engine_result(...)
            log.routing(...)
            log.discard(...)
            log.absence(...)
    """

    def __init__(self, results_dir: Path):
        results_dir = Path(results_dir)
        self._engine_results = JsonlWriter(results_dir / "engine_results.jsonl")
        self._routing = JsonlWriter(results_dir / "routing_log.jsonl")
        self._discards = JsonlWriter(results_dir / "discards.jsonl")
        self._absences = JsonlWriter(results_dir / "absences.jsonl")

    def engine_result(self, rec: EngineResult) -> None:
        self._engine_results.write(rec)

    def routing(self, rec: RoutingDecision) -> None:
        self._routing.write(rec)

    def discard(self, rec: DiscardRecord) -> None:
        self._discards.write(rec)

    def absence(self, rec: AbsenceRecord) -> None:
        self._absences.write(rec)

    def close(self) -> None:
        for w in (self._engine_results, self._routing, self._discards, self._absences):
            w.close()

    def __enter__(self) -> "CascadeLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
