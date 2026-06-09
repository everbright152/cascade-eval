"""Turn a CorpusResult into human-readable artifacts.

Writes CSV tables, a metrics.md summary, and (when matplotlib is present) the
two figures that make the findings land: confidence-vs-CER (the confident-but-
wrong scatter) and robustness deltas. Figures degrade gracefully — if
matplotlib isn't installed the tables still write and a note is logged, so the
harness never hard-fails on a plotting dependency.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

from .pipeline import CorpusResult

_HAVE_MPL = importlib.util.find_spec("matplotlib") is not None


def _fmt(x) -> str:
    return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _agg_rows(agg: dict, key_name: str) -> list[dict]:
    return [{key_name: k, **v} for k, v in sorted(agg.items())]


def _md_table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "_(none)_\n"
    head = "| " + " | ".join(cols) + " |\n"
    sep = "| " + " | ".join("---" for _ in cols) + " |\n"
    body = "".join(
        "| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |\n" for r in rows
    )
    return head + sep + body


def write_metrics_md(result: CorpusResult, path: Path) -> None:
    agg = result.aggregate
    eng = _agg_rows(agg.get("per_engine", {}), "engine")
    scr = _agg_rows(agg.get("per_script", {}), "script")
    es = _agg_rows(agg.get("per_engine_script", {}), "engine|script")

    n_decisions = len(result.decisions)
    n_absent = sum(1 for d in result.decisions if d.winner is None)
    silent = [r for r in result.robustness_rows if r.silent_failure is True]
    undetectable = [r for r in result.robustness_rows if r.silent_failure is None]

    cols = ["engine", "n_pages", "n_scored", "n_unscored", "mean_cer", "mean_wer"]
    lines = [
        "# cascade-eval — results\n",
        f"Pages routed: **{n_decisions}** · ruled unrecoverable (documented absence): "
        f"**{n_absent}**\n",
        "\n> CER/WER are reported **only** for pages with ground truth. Pages without "
        "ground truth are counted under `n_unscored` and assessed by reference-free "
        "proxies only — never assigned a fabricated CER.\n",
        "\n## CER / WER per engine\n",
        _md_table(eng, cols),
        "\n## CER / WER per script\n",
        _md_table(scr, ["script", "n_pages", "n_scored", "n_unscored", "mean_cer", "mean_wer"]),
        "\n## Per engine × script\n",
        _md_table(es, ["engine|script", "n_pages", "n_scored", "n_unscored", "mean_cer", "mean_wer"]),
        "\n## Robustness slice\n",
        f"Degraded re-runs: **{len(result.robustness_rows)}** · "
        f"**silent failures** (CER rose, confidence held): **{len(silent)}** · "
        f"undetectable via confidence (vision-LLM): **{len(undetectable)}**\n",
        _md_table(
            [r.as_dict() for r in result.robustness_rows],
            ["page", "engine", "perturbation", "delta_cer", "delta_conf", "silent_failure"],
        ),
    ]
    path.write_text("".join(lines), encoding="utf-8")


def _plot_confidence_vs_cer(result: CorpusResult, path: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [
        (s.ref_free.get("engine_confidence"), s.cer)
        for s in result.scores
        if s.scored and s.ref_free and s.ref_free.get("engine_confidence") is not None
    ]
    fig, ax = plt.subplots(figsize=(6, 5))
    if pts:
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, alpha=0.7)
    ax.axhspan(0.3, 1.0, xmin=0.0, xmax=1.0, color="red", alpha=0.05)
    ax.set_xlabel("engine confidence")
    ax.set_ylabel("CER (lower is better)")
    ax.set_title("Confidence vs. correctness\n(top-right = confident but wrong)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def _plot_robustness(result: CorpusResult, path: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_pert: dict[str, list[float]] = {}
    for r in result.robustness_rows:
        if r.delta_cer is not None:
            by_pert.setdefault(r.perturbation, []).append(r.delta_cer)
    fig, ax = plt.subplots(figsize=(6, 4))
    if by_pert:
        names = list(by_pert)
        means = [sum(v) / len(v) for v in by_pert.values()]
        ax.bar(names, means, color="steelblue")
    ax.set_ylabel("mean ΔCER (clean → degraded)")
    ax.set_title("Robustness: metric movement under perturbation")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def write_all(result: CorpusResult, results_dir: Path) -> dict:
    """Write every artifact. Returns a dict of what was written/skipped."""
    results_dir = Path(results_dir)
    figs = results_dir / "figures"
    written = {"csv": [], "figures": [], "skipped": []}

    write_csv(results_dir / "per_engine_metrics.csv",
              _agg_rows(result.aggregate.get("per_engine", {}), "engine"))
    write_csv(results_dir / "per_script_metrics.csv",
              _agg_rows(result.aggregate.get("per_script", {}), "script"))
    write_csv(results_dir / "robustness_slice.csv",
              [r.as_dict() for r in result.robustness_rows])
    written["csv"] = ["per_engine_metrics.csv", "per_script_metrics.csv", "robustness_slice.csv"]

    write_metrics_md(result, results_dir / "metrics.md")
    written["csv"].append("metrics.md")

    if _HAVE_MPL:
        figs.mkdir(parents=True, exist_ok=True)
        _plot_confidence_vs_cer(result, figs / "confidence_vs_cer.png")
        _plot_robustness(result, figs / "robustness_deltas.png")
        written["figures"] = ["confidence_vs_cer.png", "robustness_deltas.png"]
    else:
        written["skipped"].append("figures (matplotlib not installed)")

    return written


def print_summary(result: CorpusResult, written: dict) -> None:
    agg = result.aggregate.get("per_engine", {})
    n_absent = sum(1 for d in result.decisions if d.winner is None)
    silent = sum(1 for r in result.robustness_rows if r.silent_failure is True)
    print("\n=== cascade-eval summary ===")
    print(f"pages routed: {len(result.decisions)} | documented absences: {n_absent}")
    print("per-engine mean CER (scored pages only):")
    for engine, a in sorted(agg.items()):
        print(f"  {engine:<12} CER={_fmt(a['mean_cer'])} WER={_fmt(a['mean_wer'])} "
              f"(scored {a['n_scored']}/{a['n_pages']})")
    print(f"robustness: {len(result.robustness_rows)} degraded runs, {silent} silent failure(s)")
    if written.get("figures"):
        print(f"figures: {', '.join(written['figures'])}")
    for s in written.get("skipped", []):
        print(f"skipped: {s}")
