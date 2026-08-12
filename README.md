# Experiment registry - zdravkov.info

Frozen plans and results behind the research notes on [zdravkov.info](https://zdravkov.info/blog/).
The method is described in [Why I pre-register a kill-gate for every experiment](https://zdravkov.info/blog/kill-gates/).

Rules, in short: every experiment registers a plan with hypotheses and kill-gate
thresholds BEFORE any code runs; plans are frozen and amendments are appended as a
dated journal, never edited in place; only public or synthetic data is used;
results are published with the same weight whether gates pass or fire.

Plan files are in Bulgarian (the working language of the registry); each
experiment's README below states the registered gates and outcomes in English.
Published copies are redacted only where a line contained non-public detail;
every redaction is marked in place.

Hardware for all experiments: RTX 5070 Ti (16 GB VRAM), Windows 11.

| # | Experiment | Registered gates | Outcome |
|---|---|---|---|
| 001 | [PII recall by entity type](001-pii-recall-by-entity-type/) | name recall >= 0.90 (en); en-bg gap >= 5 pts; fits in 16 GB | KG1 fired (surnames 0.291); KG2 could not be pronounced (0 Cyrillic names in the bg subset); KG3 passed (1.37 GB peak) |
| 002 | [Ensemble vs GLiNER2-PII](002-ensemble-vs-gliner2/) | combined name recall >= 0.90 (en); email/phone recall >= 0.95; phone precision vs confusable numbers >= 0.99; en-bg gap < 10 pts | KG1 passed (names ~0.963, surnames 0.970); KG2a fired (phone recall 0.400); KG2b fired (phone precision 0.36); KG3 passed |
| 003 | [Blackwell quantization on my own card](003-blackwell-quantization/) | Q4_K_M throughput >= 1.3x fp16; avg embedding similarity Q4_K_M vs fp16 >= 0.90; q8_0 diagnostic only | KG1 passed (2.22x); KG2 passed (0.9835 similarity); KG3 diagnostic (1.61x, 0.9984 similarity) |
