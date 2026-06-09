"""Entrypoint for the cascade-eval harness.

Reads the corpus manifest, runs the cascade over every page, scores against
ground truth where it exists, computes the robustness slice, and writes all
result artifacts. One command, clone-to-results.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

from .engines import build_engines
from .logger import CascadeLogger
from .pipeline import PageItem, run_corpus
from .report import print_summary, write_all

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"config not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f)


def _truthy(v: str | None) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def load_pages(config: dict) -> list[PageItem]:
    """Build PageItems from data/manifest.csv. The robustness subset is the
    first N pages (config.robustness.subset_size) that have ground truth."""
    paths = config["paths"]
    manifest = Path(paths["manifest"])
    if not manifest.exists():
        sys.exit(
            f"no manifest at {manifest}.\n"
            "Assemble the corpus first (PLAN.md Step 1): drop page images in "
            f"{paths['images']}/, ground truth in {paths['ground_truth']}/, and "
            "list them in manifest.csv. See data/ for the expected columns."
        )
    from PIL import Image
    import numpy as np

    images_dir = Path(paths["images"])
    subset_budget = config.get("robustness", {}).get("subset_size", 5)
    pages: list[PageItem] = []
    subset_used = 0
    with manifest.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            img_path = images_dir / row["image"]
            arr = np.array(Image.open(img_path).convert("RGB"))
            gt_path = (row.get("gt_path") or "").strip()
            gt_text = Path(gt_path).read_text(encoding="utf-8") if gt_path else None
            in_subset = gt_text is not None and subset_used < subset_budget
            if in_subset:
                subset_used += 1
            pages.append(
                PageItem(
                    page=Path(row["image"]).stem,
                    image=arr,
                    script=row["script"],
                    language=row["language"],
                    ground_truth=gt_text,
                    is_handwritten=_truthy(row.get("is_handwritten")),
                    in_robustness_subset=in_subset,
                )
            )
    return pages


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cascade-eval",
        description="Evaluate an OCR/HTR cascade and surface where it silently fails.",
    )
    p.add_argument("command", nargs="?", default="run", choices=["run"],
                   help="run: full pipeline over the manifest")
    p.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG,
                   help=f"path to config.yaml (default: {DEFAULT_CONFIG})")
    p.add_argument("--no-llm", action="store_true",
                   help="disable the vision-LLM fallback engine even if a key is set")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    engines = build_engines(config, allow_llm=not args.no_llm)
    live = [n.value for n, e in engines.items() if e.is_available]
    print(f"cascade-eval :: engines configured={[n.value for n in engines]} live={live}")

    pages = load_pages(config)
    print(f"corpus: {len(pages)} pages from {config['paths']['manifest']}")

    results_dir = Path(config["paths"]["results"])
    with CascadeLogger(results_dir) as logger:
        result = run_corpus(config, engines, logger, pages)

    written = write_all(result, results_dir)
    print_summary(result, written)
    print(f"\nartifacts in {results_dir}/ (logs: routing_log.jsonl, discards.jsonl, absences.jsonl)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
