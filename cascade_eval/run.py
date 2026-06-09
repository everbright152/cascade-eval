"""Entrypoint for the cascade-eval harness.

Step 0 scaffold: wires up the CLI surface and config loading. The actual
pipeline stages (corpus -> cascade -> metrics -> report) are filled in by
later steps; for now `run` reports what it *would* do so the harness is
importable and the CLI contract is locked.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"config not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cascade-eval",
        description="Evaluate an OCR/HTR cascade and surface where it silently fails.",
    )
    p.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["run", "robustness", "report"],
        help="run: full pipeline | robustness: degraded-input slice | report: rebuild tables/figures",
    )
    p.add_argument(
        "-c", "--config", type=Path, default=DEFAULT_CONFIG,
        help=f"path to config.yaml (default: {DEFAULT_CONFIG})",
    )
    p.add_argument(
        "--no-llm", action="store_true",
        help="disable the vision-LLM fallback engine even if a key is set",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    # Step 0: prove the wiring works. Later steps replace this block.
    print(f"cascade-eval :: command={args.command}")
    print(f"  config: {args.config}")
    print(f"  engines enabled: {[k for k, v in config['engines'].items() if v.get('enabled')]}")
    print(f"  thresholds: {config['routing']}")
    print("  [scaffold] pipeline not yet implemented — see PLAN.md steps 1-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
