# 010 - How much does an adapter forget?

Registered 2026-10-30, executed 2026-10-30. Frozen plan: [plan.md](plan.md)
(Bulgarian; published copy, no threshold or result altered).

**Question.** Does fine-tuning a model with LoRA on a narrow domain cost
measurable general capability, and does that cost scale with adapter rank and
training steps (H1), or is there a configuration that gains real domain
accuracy while keeping general-task retention above a registered floor (H2)?

**Model and data.** Qwen2.5-3B-Instruct, 4-bit base, LoRA on the q/k/v/o
projections only. Domain task: `nguha/legalbench`, config `overruling`
(binary legal classification, 1915/479 train/test split carved from the
dataset's 2394-example `test` split). General task: 120 randomly sampled
`cais/mmlu` questions, fixed across every cell. RTX 5070 Ti (16 GB), torch
2.11.0+cu128, PEFT.

**Grid.** LoRA rank {4, 16, 64} x training steps {50, 200, 500}, 9 cells, one
run per cell. See [02_run_grid.py](run/02_run_grid.py), full numbers in
[results_grid.json](results_grid.json).

**Registered gates and verdicts.**
- KG1 (H1: forgetting grows with rank/steps, threshold >=5pp retention drop
  between the smallest and largest cell): **NOT MET.** Retention rose from
  100.0% to 109.1% between the smallest and largest cell - the opposite
  direction from the hypothesis. H1 is rejected.
- KG2 (H2: a working point exists, domain gain >10pp at retention >95%):
  **MET** in 8 of 9 cells.

**Why this needed a second look before publishing the numbers.** Every single
cell showed retention at or above 100% - never once below the baseline. Under
pure sampling noise roughly half the cells should have landed below. Three
checks, in order, before trusting the result:

1. **Parsing artifact?** Compared raw generated text from the baseline model
   against the largest-gain cell on the same 120 questions. Both answer in a
   clean one-letter format almost every time (1/120 and 0/120 unparseable).
   Not a parsing issue.
2. **Non-deterministic loading/quantization?** Reloaded the unmodified base
   model twice, independently, and compared raw answers on the same 120
   questions. Zero differences. Loading and 4-bit quantization are fully
   deterministic here.
3. **Missing seed control.** The original grid trains each cell exactly once,
   with no fixed seed for the LoRA adapter's random init - a real design gap.
   Re-ran the smallest and largest cell under two new explicit seeds each
   ([05_check_seed_variance.py](run/05_check_seed_variance.py),
   [seed_variance_check.json](seed_variance_check.json)). The smallest cell's
   retention now lands on both sides of 100% (98.5%, 104.5%) - consistent with
   noise around a near-zero effect, and it explains why the single original
   run happened to land on exactly 100.0%. The largest cell stayed above
   baseline in both reseeded runs (110.6%, 106.1%), for three out of three
   independent trainings total. That is a replicated effect, not a one-off.

**Headline finding.** No forgetting was observed in the tested range.
Stronger LoRA fine-tuning on a narrow, unrelated legal-classification task
appears to modestly and reproducibly *improve* performance on an unrelated
general multiple-choice benchmark, rather than degrade it. No mechanism is
confirmed - a plausible, untested hypothesis is that LoRA training toward a
terse, forced answer format transfers some general "commit to an answer"
behavior that happens to help multiple-choice accuracy too, independent of
actual knowledge. A control run (LoRA fine-tuned on irrelevant/random data,
to isolate whether the effect comes from Overruling specifically or from any
LoRA training at all) was not run here - flagged as a follow-up, not part of
this registered result.

**Limitations.** Single run per cell in the main 3x3 grid, no seed control
(partially checked after the fact on the two extreme cells only). The general
task is 120 questions spread across 57 MMLU subjects - too coarse to detect
fine, subject-distributed forgetting even if it exists. One model family, one
narrow binary-classification domain task; nothing here generalizes to longer
generative domain tasks or other model families.

Full tables and every diagnostic script's raw output:
[results.md](results.md), [results_grid.json](results_grid.json),
[diagnostic_parsing.json](diagnostic_parsing.json),
[seed_variance_check.json](seed_variance_check.json).
