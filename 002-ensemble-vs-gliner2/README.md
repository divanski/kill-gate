# 002 - Ensemble of standard components vs GLiNER2-PII

Registered 2026-07-27, executed 2026-07-27. Frozen plan: [plan.md](plan.md)
(Bulgarian; published copy with marked redactions of non-public lines only -
no threshold or result was altered).

**Question.** The surname failure in experiment 001 dictated this one: does an
ensemble of standard open components (Presidio 2.2.358 orchestration,
Davlan/xlm-roberta-base-ner-hrl on transformers 4.47.1, spaCy 3.7.5 +
xx_ent_wiki_sm, lingua-language-detector, checksum-validated national ID
recognizers, boosted phone recognizer) recover what the single small model
misses?

**Data.** The same frozen sample as experiment 001. Pinned CPU environment
(torch 2.5.1 has no Blackwell support); ~60ms/example.

**Registered gates and verdicts.**
- KG1: combined name recall >= 0.90 (en). PASSED: given names 0.957,
  surnames 0.970 (vs 0.291 for the single model).
- KG2a: email/phone recall >= 0.95 (en). FIRED: email 1.000, phone 0.400
  (vs 0.995 for the single model).
- KG2b: phone precision vs confusable numeric types >= 0.99. FIRED: 0.362 (en)
  / 0.354 (bg) - phone predictions land on cards, IDs and social numbers
  almost twice as often as on actual phones, on this dataset's synthetic
  formats.
- KG3: en-bg name gap < 10 points. PASSED: under 1 point.

**Conclusion.** Neither configuration dominates: the ensemble fixes names, the
single model wins on phones, dates and cards. All numbers describe this
dataset's synthetic formats, not any live traffic.

**Notable instrument limitation:** checksum-validated ID recognizers cannot be
recall-tested on synthetic data - random digit strings fail validation by
design (they fired rarely, at precision 1.000).

Full tables: [metrics_stack.json](metrics_stack.json). Evaluation code:
[03_eval_stack.py](03_eval_stack.py) (the ensemble assembly itself is built
from the standard components listed above and is not included).
