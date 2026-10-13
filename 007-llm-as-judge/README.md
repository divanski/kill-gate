# 007 - An honest look at LLM-as-a-judge

Registered 2026-10-13, executed 2026-10-13. Frozen plan: [plan.md](plan.md).

**Question.** Does a small, local LLM used as a judge agree with human ratings
well enough to be trusted as a proxy? And how stable is its own judgement when
asked about the same thing more than once?

**Setup.** `nvidia/HelpSteer2` (public, CC-BY-4.0), validation split, 150
randomly sampled items (seed 42). Judge: `gpt-oss:latest`, local via Ollama on
an RTX 5070 Ti. Each item rated 5 times independently (temperature 0.7) on
the same 0-4 helpfulness scale HelpSteer2's own human annotators used.

**Registered gates and verdicts.**
- KG1: Spearman rho between judge and human helpfulness >= 0.5. FIRED (H1
  rejected): actual rho = 0.395 (p < 0.0001), significant but weak.
- KG2: mean standard deviation across the 5 repeats <= 0.75. PASSED, with
  margin: actual mean std = 0.386.
- KG3 (diagnostic): correlation between the judge's own instability and its
  error. Weak but real signal: rho = 0.231 (p = 0.0044).

**The number that isn't obvious from the gates alone.** Precision and
accuracy are different things, and this experiment separates them cleanly.
54 of 150 items (36%) got the identical score across all 5 repeats - the
judge is precise. But its rounded mean matched the human score on only 36%
of items, with a mean absolute error of 0.97 on a 5-point scale - precision
did not buy accuracy.

**A concrete example.** Prompt: list 10 video games related to Bruce Lee in
a table. Response: a well-formatted table with 10 titles, years and roles.
Human rating: 4 (extremely helpful). Judge, all 5 repeats: 0, 0, 0, 0, 0
(not helpful at all) - confidently and consistently wrong, with zero
self-reported uncertainty to flag it.

**Score distributions (human vs judge, n=150):**

| Score | Human count | Judge (rounded mean) count |
|---|---|---|
| 0 | 8 | 19 |
| 1 | 14 | 7 |
| 2 | 9 | 42 |
| 3 | 50 | 52 |
| 4 | 69 | 30 |

**What it does not show.** One model (`gpt-oss`), one sample size, one
rubric wording - does not claim a larger model or a different prompt would
behave the same way. Only the helpfulness dimension was tested, not
HelpSteer2's other four (correctness, coherence, complexity, verbosity).

Full data: [metrics.json](metrics.json), experiment code in [run/](run/).
