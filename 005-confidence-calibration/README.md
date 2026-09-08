# 005 - How wrong is a model's confidence?

Registered 2026-09-08, executed 2026-09-08. Frozen plan: [plan.md](plan.md).

**Question.** Does a modern, high-capacity classifier produce softmax
confidence values that match its actual accuracy, or is it systematically
overconfident? And if so, does temperature scaling (Guo et al. 2017, arXiv
1706.04599), a single learned scalar, fix it without touching accuracy?

**Setup.** `edadaltocg/resnet34_cifar100`, a public ResNet-34 checkpoint
(MIT licence, reported 79.78% test accuracy), evaluated on the official
CIFAR-100 test set (10,000 images), split in half: `dev` (5,000, fits the
temperature) and `eval` (5,000, final numbers only). Expected Calibration
Error (ECE), 15 bins, computed before and after scaling. RTX 5070 Ti.

**Registered gates and verdicts.**
- KG1: ECE on `eval` before calibration under 3%. DID NOT FIRE (H1
  confirmed): actual ECE was 7.87%, more than double the threshold.
- KG2: relative ECE reduction from temperature scaling under 30%. DID NOT
  FIRE (H2 confirmed): actual reduction was 63.7% (7.87% -> 2.85%).
- KG3 (diagnostic): top-1 accuracy unchanged by scaling. PASSED: 80.24%
  before and after, as the maths of a monotonic rescaling predicts.

**Why this is convincing, not a coin flip.** Both gates cleared their bar
by more than double, not by a thin margin. The top confidence bin
(93.3-100%) held 70.1% of all predictions before scaling, averaging 99.11%
stated confidence against 94.27% actual accuracy - a 4.84 point gap sitting
inside the bin most predictions live in, not a tail case. After scaling
that gap shrank to 0.88 points.

**What it does not show.** One checkpoint, one architecture family, one
dataset. The `dev`/`eval` split came from halving the official test set,
not from an independent held-out validation set, because the checkpoint
was already trained on the full training split - a documented limitation,
not a hidden one. Says nothing about calibration in large language models,
a separate and less settled problem (Liu et al. 2025, arXiv 2503.15850).

Full data: [metrics.json](metrics.json), experiment code in [run/](run/).
