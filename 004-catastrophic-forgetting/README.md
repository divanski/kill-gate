# 004 - Catastrophic forgetting, measured

Registered 2026-08-14, executed 2026-08-14. Frozen plan: [plan.md](plan.md).

**Question.** The smallest reproducible demonstration of catastrophic
forgetting: sequential training of one small network on two tasks. Does
plain fine-tuning forget Task A while learning Task B, and does Elastic
Weight Consolidation (EWC, Kirkpatrick et al. 2017, arXiv 1612.00796)
actually protect it, as the original paper claims?

**Setup.** Permuted MNIST (Task A = MNIST, Task B = the same images with a
fixed random pixel permutation). MLP 784-400-400-10, Adam, 5 epochs per
task. EWC: diagonal Fisher information from 1,000 samples of Task A, λ=1000.
Public MNIST data only. RTX 5070 Ti, 26.3 seconds total wall clock.

**Registered gates and verdicts.**
- KG1: Task A accuracy drop under plain sequential fine-tuning >= 20
  percentage points. FIRED (H1 rejected): actual drop was 12.3 points
  (97.88% -> 85.55%), real forgetting but below the registered bar.
- KG2: EWC recovers >= 50% of the lost accuracy. PASSED, by a wide margin:
  98.0% of the drop recovered (accuracy back up to 97.63%).

**Why KG1 fired.** Permuted MNIST is the easy end of continual-learning
benchmarks - both tasks share the same output space (10 shared classes),
unlike harder setups such as Split MNIST with disjoint output heads, where
forgetting tends to be more severe. Five epochs with Adam on a small MLP
was not an aggressive enough regime to clear the registered 20-point bar,
even though real, measurable forgetting occurred. The threshold was not
adjusted after seeing the result.

**Why KG2 is convincing.** 98% recovery from a diagonal Fisher estimate
built from only 1,000 samples is a clean result. EWC's Task B accuracy
(97.43%) was nearly identical to the unprotected run's (97.84%) - in this
setup, protecting Task A did not cost noticeable Task B performance.

Full data: [metrics.json](metrics.json), experiment code in [run/](run/).
