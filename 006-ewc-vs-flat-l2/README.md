# 006 - Does EWC's Fisher weighting actually earn its keep against a flat L2 control?

Registered 2026-09-16, executed 2026-09-16. Frozen plan: [plan.md](plan.md).

**Question.** Experiment 004 found EWC recovers 98.0% of the accuracy lost
to catastrophic forgetting on Permuted MNIST. EWC's Fisher-weighted penalty
adds real complexity: a diagonal Fisher information estimate from 1,000
samples before every new task. Does that complexity earn anything over a
flat (uniform, non-Fisher-weighted) L2 penalty of comparable "budget"?

**Setup.** Same infrastructure as Experiment 004: Permuted MNIST, MLP
784-400-400-10, Adam, 5 epochs per task, seed 42. A flat L2 penalty
(lambda/2 * sum((theta - theta_A)^2), no Fisher weighting) swept across six
values (0.01 to 1000, log scale), registered before running. Fair-budget
rule, fixed in advance: report the L2 setting whose Task B accuracy cost is
closest to (but under) 1 percentage point, matching EWC's own 0.41pp cost.

**Registered gate and verdict.**
- KG1 (best-budget flat L2 within 3pp of EWC's Task A accuracy -> H
  rejected): FIRED, literally - the cheapest flat L2 setting (lambda=0.01)
  landed 0.86pp from EWC (96.77% vs 97.63%). But no lambda in the sweep
  actually met the 1pp budget: the cheapest option still cost 2.70pp on
  Task B, over 6x EWC's cost. `fair_pick_in_budget: false` in metrics.json
  flags this explicitly rather than silently reporting a clean pass.

**The number that tells the real story.** At lambda=10, flat L2 matches
EWC's Task A accuracy almost exactly (97.63% both) - but Task B accuracy
collapses to 18.42%, against EWC's 97.43%. Flat L2 can hit EWC's retention
number, but only by nearly freezing the network rather than protecting
what matters and leaving the rest free to adapt.

**Full numbers.**

| Condition | Task A accuracy | Task B accuracy | Task B cost vs plain |
|---|---|---|---|
| After Task A | 97.88% | - | - |
| Plain (no protection) | 85.55% | 97.84% | - |
| EWC (Fisher, lambda=1000) | 97.63% | 97.43% | 0.41pp |
| Flat L2, lambda=0.01 | 96.77% | 95.14% | 2.70pp |
| Flat L2, lambda=0.1 | 97.36% | 88.62% | 9.22pp |
| Flat L2, lambda=1 | 97.38% | 64.85% | 32.99pp |
| Flat L2, lambda=10 | 97.63% | 18.42% | 79.42pp |
| Flat L2, lambda=100 | 97.90% | 10.53% | 87.31pp |
| Flat L2, lambda=1000 | 97.90% | 10.16% | 87.68pp |

**What it does not show.** No lambda in this six-point sweep lands in the
registered 1pp fair-budget zone, so the comparison at the honest budget
point does not exist yet - a finer sweep near 0.01-0.1 is the natural
follow-up, not run here because it was not pre-registered. Does not test
optimizer-internal claims (e.g. Adam's persistent first-moment state) -
both EWC and flat L2 here act as loss penalties, not modifications to
optimizer state.

Full data: [metrics.json](metrics.json), experiment code in [run/](run/).
