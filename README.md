# cascade-eval

A small, runnable harness for evaluating a multilingual **OCR/HTR cascade** —
and, more to the point, for finding where it **silently fails**. It routes each
page between two or more engines with a legible, logged decision; reports CER/WER
per engine and per script; degrades inputs to show how the metrics move; and
treats discards and *documented absence* as first-class output rather than
debugging noise.

Built for the QuarterMill trial task. The emphasis is on **honest measurement**,
not a high score: where there is no ground truth, the harness says so instead of
manufacturing a number.

---

## Quickstart (clone → results)

```bash
git clone <repo-url> cascade-eval
cd cascade-eval

# Installs the one system dependency (Tesseract + eng/ara language packs)
# and the Python environment (EasyOCR/PyTorch, jiwer, matplotlib, ...).
./setup.sh

# Run the full cascade over the corpus (no API key needed).
python -m cascade_eval.run --no-llm        # or: .venv/bin/python -m cascade_eval.run --no-llm
```

Results land in `results/`. Read `results/metrics.md` first.

> **Timing note:** `setup.sh` installs Tesseract via `apt`/`brew` (fast) and
> EasyOCR, which pulls PyTorch (~1 GB) and downloads its recognition models on
> the **first** run. First run can take several minutes; subsequent runs are quick.

### If `setup.sh` can't use `sudo`

Tesseract is a system binary. Install it yourself, then run `setup.sh` (it skips
what's already present):

```bash
sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-ara
```

### Optional: vision-LLM fallback

A third engine (Claude vision) handles hard low-resource / handwritten pages.
It's **optional** and the harness degrades gracefully without it. To enable:

```bash
cp .env.example .env      # add ANTHROPIC_API_KEY=...
python -m cascade_eval.run   # drop --no-llm
```

---

## What it produces (`results/`)

| File | What it is |
|---|---|
| `metrics.md` | Human-readable summary: CER/WER per engine, per script, per engine×script, and the robustness slice. **Start here.** |
| `per_engine_metrics.csv`, `per_script_metrics.csv` | The same tables as CSV. |
| `robustness_slice.csv` | Clean→degraded deltas per (page, engine, perturbation) + the silent-failure flag. |
| `routing_log.jsonl` | One record per page: winner, candidates considered, thresholds in force, and a human-readable reason. |
| `discards.jsonl` | Every rejected candidate, with the reason it lost. |
| `absences.jsonl` | Every page/region ruled unrecoverable, with a flagged best-effort guess. |
| `engine_results.jsonl` | Every engine's raw attempt (including engines that couldn't run). |
| `figures/confidence_vs_cer.png` | Confidence vs. CER scatter — the "confident but wrong" view. |
| `figures/robustness_deltas.png` | Mean ΔCER per perturbation. |

Everything in `results/` is regenerated from scratch on each run.

---

## How it works

```
corpus (manifest) → preprocess → script detect → CASCADE → score → robustness → report
                                                    │
                                       ┌────────────┴─────────────┐
                                       │ route between engines,    │
                                       │ log decision + discards   │
                                       └───────────────────────────┘
```

**Engines (chosen to fail *differently*, so routing is meaningful):**
- **Tesseract** — classical baseline; broad script coverage; exposes **per-word confidence**.
- **EasyOCR** — neural detector+recognizer; fails differently from Tesseract; per-box confidence.
- **Vision-LLM (Claude, optional)** — fallback for low-resource/handwritten pages. Reports **no confidence** (the API has no logprobs) — by design, so the router judges it on reference-free proxies, not a self-reported number.

**Routing (legible, threshold-driven — all knobs in `config.yaml`):**
1. Tesseract first; accept if `confidence ≥ tau_high` **and** a lexical signal `≥ delta`
   (dictionary-hit where a wordlist exists, else script-consistency).
2. Else EasyOCR; same gate (skipped + logged when it has no model for the script).
3. Else, for low-resource / handwritten pages, escalate to the vision-LLM.
4. Else pick the best-of-available by combined reference-free proxy — but only if it clears `tau_min`.
5. Else the page is ruled **unrecoverable** → an `AbsenceRecord` (documented absence).

**Reference-free quality** (for pages with no ground truth): engine confidence,
dictionary hit-rate, and **script-consistency** (needs no external resource — works
for low-resource languages *and* catches an engine emitting fluent gibberish in the
wrong script).

See `RATIONALE.md` for findings, failure analysis, and how this detects silent
degradation in production.

---

## Corpus

19 page images across three scripts, documented in `data/provenance.json`:

| Script | Pages | Notes |
|---|---|---|
| Latin (English) | `eng_001..008` | historic printed pages; `eng_007/008` are clean cropped regions **with ground truth** |
| Arabic (RTL) | `ara_001..006` | printed + handwritten manuscripts |
| Jawi (low-resource) | `jawi_001..005` | Malay/Arabic-script manuscripts + 1 print |

**Ground truth is deliberately thin.** Only the two clean English crops were
hand-transcribed (high confidence). The Arabic full pages were *not*
author-transcribed — fabricating reference text would be exactly the
"manufactured number" the task warns against — so they are honestly flagged
`has_GT=false` and assessed by reference-free proxies only. This is intentional:
the corpus exercises **both** the scored path and the no-answer-key path.

`data/manifest.csv` is the entry point (`image, script, language, gt_path,
is_lowresource, is_handwritten`); the harness iterates its rows.

---

## Configuration

All routing thresholds live in `config.yaml` and are echoed into every routing
record, so the policy stays auditable:

- `tau_high` (0.85) — accept a candidate outright at/above this confidence
- `tau_min` (0.50) — below this, a candidate cannot win → may trigger documented absence
- `delta_dict` (0.40) — minimum lexical confirmation to trust a high-confidence result
- `robustness.perturbations` — blur / skew / lowres parameters

---

## Tests

```bash
.venv/bin/python -m tests.smoke_engines     # which engines are live on this machine
for t in test_schemas_roundtrip test_quality test_router test_metrics \
         test_robustness test_pipeline_report; do .venv/bin/python -m tests.$t; done
# or, with dev extras installed:  pytest -q
```

The logic tests run **without** any OCR engine installed (synthetic inputs),
which is how the cascade, metrics, and robustness logic are verified
independently of the engines.

---

## Reproducibility notes

- One system dependency (Tesseract); everything else is pinned Python.
- Engines self-report availability and **degrade gracefully** — a missing engine
  or API key is logged, never crashed. With no engine installed at all, every
  page becomes a documented absence (nothing is fabricated).
- Outputs are regenerated each run; logs flush per-write so they survive a crash.
