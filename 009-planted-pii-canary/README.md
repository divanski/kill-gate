# 009 - Planted-PII canary

Registered 2026-10-29, executed 2026-10-29. Frozen plan: [plan.md](plan.md)
(Bulgarian; published copy with marked redactions of non-public lines only -
no threshold or result was altered).

**Origin.** This plan operationalizes an idea from a reader comment (dev.to
user "shuttlewright") on the Research note "The masking layer matters more
than the model": aggregate recall hides per-type silent misses, and a cheap
way to surface them is to continuously run synthetic "planted" PII through
the stack and watch recall per type, not just the average.

**Question.** Can a small set of synthetic planted PII, inserted into neutral
carrier sentences, (a) show the kind of per-type recall gap an aggregate
number hides, and (b) actually detect an injected regression - proving the
method is a useful continuous canary, not just a one-off number?

**Data.** 240 English + 280 Bulgarian synthetic sentences (Faker, seed
909/910), each with one or more planted PII spans labeled at generation time.
Not derived from any annotated dataset or real registry - see
[01_generate_planted.py](01_generate_planted.py) and [data/](data/).

**Registered gates and verdicts.**
- KG1: spread between best- and worst-performing type >= 20 points. PASSED:
  30.0pp (en, EMAIL 1.000 vs CREDITCARDNUMBER 0.700), 100.0pp (bg, SURNAME
  1.000 vs CREDITCARDNUMBER 0.000).
- KG2: targeted recall drop >= 30pp when the phone recognizer is disabled,
  unrelated-type drift < 5pp. PASSED: TELEPHONENUM dropped 75.0pp in both
  languages, every other type moved 0.0pp.

**Headline finding (outside the registered hypotheses, but the most direct
answer to the original comment).** Credit-card recall on Bulgarian is not
low - it is exactly 0.000. Inspecting the recognizer registry directly shows
why: the credit-card pattern recognizer is registered only under the `en`
language, never added for `bg` the way the national-ID recognizers explicitly
were. No error, no warning, no failing test - a card number mentioned in a
Bulgarian sentence passes through completely unflagged, because nothing had
tested that combination before. This is the exact failure mode the comment
described: not a visible bug, a silent one.

**Second, smaller finding.** The English-side credit-card misses (0.700, not
1.000) have a different cause: Faker's default card generator produces
several lengths (12/14/15/19 digits), and the recognizer's brand-specific
regexes only cover the common ones. Same shape of problem - silent until you
test that exact format - different root cause.

Full tables: [results_canary.json](results_canary.json). Evaluation code:
[02_eval_canary.py](02_eval_canary.py) (the stack itself is assembled from the
standard, open components listed in plan.md and is not included).

**Next.** "Continuously run" here means a reproducible, seeded script, not
yet a scheduled job - a real recurring canary run is planned as a follow-up,
by the author's own decision in plan.md.
