# 001 - PII recall by entity type

Registered 2026-07-27, executed 2026-07-27. Frozen plan: [plan.md](plan.md) (Bulgarian).

**Question.** How much does a small open PII model (GLiNER2-PII,
fastino/gliner2-privacy-filter-PII-multi, 0.3B) miss, per entity type, and does
Bulgarian (Cyrillic) text hurt detection?

**Data.** ai4privacy/pii-masking-openpii-1m, first 2,000 examples per language
(en, bg) in native order. The sample is reproducible with
[run/01_sample.py](run/01_sample.py); the data itself is not redistributed here.

**Registered gates and verdicts.**
- KG1: name recall >= 0.90 (en) or the "small model suffices out of the box"
  claim dies. FIRED: given names 0.850, surnames 0.291.
- KG2: en-bg name recall gap >= 5 points confirms the Cyrillic-weakness
  hypothesis. COULD NOT BE PRONOUNCED: a validity check found 100% of the
  3,410 name values in the bg subset are Latin-script; the instrument does not
  contain the stimulus. (Numerically the gap was under 2 points, but that
  compares Latin names in Cyrillic context vs Latin context.)
- KG3: must fit in 16 GB VRAM. PASSED: 1.37 GB peak, ~55 examples/s.

**Selected numbers (relaxed span overlap, en):** EMAIL 1.000, TELEPHONENUM
0.995, DATE 0.983, GIVENNAME 0.850, SURNAME 0.291, TITLE 0.131. Full tables:
[metrics.json](metrics.json).

**Notable:** 63% of missed surnames appear nowhere in predictions (probe over
300 examples); street numbers get folded into street spans; numeric ID types
confuse each other (precision 0.4-0.5).
