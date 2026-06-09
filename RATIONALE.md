# Rationale

A short account of what this harness measures, what it found, where the cascade
fails, and how I would detect silent degradation in production. Numbers below are
from the committed two-engine run (Tesseract + EasyOCR) over the 19-page corpus.

## What I built and why

The brief is not "get a high OCR score" — it's "tell us, rigorously, where an
OCR/HTR cascade is reliable and where it silently fails." So the design choices
all serve *measurement and provenance*, not accuracy:

- **Two engines that fail differently.** Tesseract (classical, per-word
  confidence) and EasyOCR (neural, per-box confidence). A cascade between two
  redundant engines learns nothing; these two disagree in useful ways. An
  optional vision-LLM adds a third, qualitatively different reader for the
  hardest pages.
- **A legible router, not a black box.** Every threshold (`tau_high`, `tau_min`,
  `delta_dict`) lives in `config.yaml` and is written into every routing record.
  You can read exactly why a page went where.
- **Discards and absence as first-class output.** Every losing candidate and
  every unreadable page is a structured record with a reason — not a log line you
  have to grep for.

## What the run found

**Routing genuinely discriminates.** Across 19 pages the router chose Tesseract
14 times and EasyOCR 5 times, logging 19 discards with reasons. It is not a
rubber stamp for one engine.

**Per-engine accuracy diverges sharply on the gradeable pages.** On the two
ground-truth English pages:

| engine | CER | WER |
|---|---|---|
| EasyOCR | 0.10 | 0.81 |
| Tesseract | 0.61 | 1.00 |

EasyOCR is far better here — these are stylized display/banner fonts, where
Tesseract's classical pipeline degrades badly. The high *WER* (even for EasyOCR)
is itself a measurement lesson: with only a handful of reference words, one
wrong token swings WER enormously, and we keep scoring case- and
punctuation-sensitive on purpose (a `.`→`-` is a real OCR error). **CER is the
trustworthy signal at this corpus size; WER is reported but noisy.** I left the
numbers ugly and explained them rather than cosmetically lowering them.

## Where the cascade fails

1. **Stylized / display typography.** Tesseract collapses on banner fonts
   (`CHRONICLING` → `CHROHIGLIHA`). Routing partly rescues this by preferring
   EasyOCR, but only because we *measure* the failure.
2. **Low-resource scripts have no safety net in the classical engines.** EasyOCR
   has no Coptic model at all and weak Jawi support; Tesseract leans on the
   Arabic pack for Jawi. Without ground truth we can't put a CER on these — see
   below — so the cascade's confidence here is *unvalidated*, which is itself the
   finding.
3. **Confidence is not correctness — and the robustness slice proves it.** When
   we degrade inputs, two cases are silent failures: `eng_008/tesseract/blur`
   (CER rose +0.52 while confidence *rose* +0.26) and `eng_007/easyocr/skew`
   (CER +0.29, confidence +0.06). In both, the engine got worse and *more*
   confident — the exact pattern that is invisible in production.
4. **A single confidence threshold cannot police both engines.** Four degraded
   Tesseract runs returned *no* confidence at all (no confident tokens), so a
   confidence-based silent-failure check is undefined for them. The harness flags
   these `undetectable` rather than scoring them as fine.

## Confidence vs. correctness

This distinction is wired into the data model, not bolted on. `EngineResult`
separates what a model *thinks* (`confidence`, per-token) from whether it is
*right* (CER, computed only against ground truth). The robustness slice joins the
two and flags `silent_failure = (CER rose materially) AND (confidence held)`. The
`confidence_vs_cer.png` scatter makes the danger zone (high confidence, high CER)
visible at a glance. The vision-LLM is the limiting case: the API exposes no
confidence at all, so we record `confidence = None` and refuse to invent one —
its quality must be judged by reference-free proxies.

## Measuring quality with no answer key

Most of this corpus (and, per the brief, most real material) has no ground truth
and never will. Standard CER/WER is undefined there. The harness computes three
**reference-free proxies** instead, and reports *which could even be computed*:

- **engine confidence** — cheap, but (as shown) can lie.
- **dictionary hit-rate** — strong for well-resourced languages; `None` when no
  wordlist exists (the low-resource case — flagged, never faked).
- **script-consistency** — fraction of letters in the *expected* script. Needs no
  external resource, so it still works for Jawi/Coptic, and it catches an engine
  emitting fluent text in the *wrong* script (a common, confident failure).

These don't replace CER — they're explicitly labeled estimates. But they let the
cascade *route* and *triage* without ground truth, and they let a reviewer see
the basis for every no-GT decision.

**Hard page vs. degraded page.** A genuinely hard but clean page tends to keep
*high script-consistency* with *moderate confidence* (the content is real,
recognition is uncertain). A degraded page shows *falling* proxies across the
board, and — the tell — the robustness slice shows its metrics moving under
perturbation while a hard-but-clean page is comparatively stable. Comparing the
clean reference-free profile against the degraded one is how I separate "hard"
from "broken."

## Surfacing documented absence

When no candidate clears `tau_min`, the page is not silently dropped: it becomes
an `AbsenceRecord` carrying the reason, the region (whole-page here; per-region is
the natural extension), and an explicitly low-confidence best-effort guess. A
downstream reader gets *what is missing and why*, in a structured, queryable
form. With no engine installed at all, all 19 pages become documented absences —
which is the correct, honest behavior, not a crash.

## How I'd detect silent degradation in production

Silent degradation is a *distribution shift you can't see in the output*. With no
ground truth at serving time, I would monitor the reference-free signals the
harness already computes and alert on their *movement*, not their absolute value:

1. **Track the proxy distribution per script over time** (confidence,
   script-consistency, dictionary-hit). A drop in mean script-consistency for a
   script, with confidence holding, is the production analogue of the silent
   failures found here — alert on it.
2. **Watch confidence–quality decoupling.** Rising confidence with falling
   reference-free quality is the signature of "confident but wrong." This is the
   robustness-slice flag, run continuously on live traffic.
3. **Treat the absence rate as a first-class SLO.** A spike in documented
   absences (or in low-`tau_min` near-misses) for a script flags an input-quality
   or model regression before any human notices wrong text.
4. **Periodic shadow scoring on a small pinned, ground-truthed slice** to anchor
   the proxies to real CER and re-calibrate thresholds.

## What I'd instrument / do next (out of time-box)

- **Per-region absence and confidence**, not just per-page — bounding boxes of
  unrecoverable glyphs so downstream readers see exactly which spans are missing.
- **Confidence calibration** (reliability diagrams per engine/script) so the
  silent-failure threshold is principled rather than a fixed tolerance.
- **More ground truth on a small, representative slice** — including a clean
  printed-body page — to make CER/WER less sample-noisy and to validate the
  reference-free proxies against real error rates.
- **An LM-perplexity proxy** (scaffolded, off by default) as a fourth
  reference-free signal, especially for scripts with no wordlist.
- **A real third engine in the routed path** (vision-LLM) on the low-resource
  pages, with cost-aware escalation.

## Honesty ledger

- 2 of 19 pages have ground truth; CER/WER is reported only for those. The other
  17 are reference-free only, clearly labeled — no fabricated numbers.
- Provenance sources are marked "unverified" where I could not confirm them.
- WER is reported but flagged as noisy at this corpus size; CER is the headline.
- The corpus was kept deliberately small with clean measurement, per the brief.
