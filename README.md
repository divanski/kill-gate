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
| 004 | [Catastrophic forgetting, measured](004-catastrophic-forgetting/) | Task A accuracy drop >= 20pp under plain fine-tuning; EWC recovers >= 50% of the drop | KG1 fired (12.3pp drop, below the bar); KG2 passed (98.0% recovered) |
| 005 | [How wrong is a model's confidence?](005-confidence-calibration/) | ECE before calibration < 3%; relative ECE reduction from temperature scaling < 30%; accuracy unchanged (diagnostic) | KG1 did not fire (ECE 7.87%); KG2 did not fire (63.7% reduction); KG3 passed (80.24% before and after) |
| 006 | [EWC vs flat L2 control](006-ewc-vs-flat-l2/) | best-budget flat L2 within 3pp of EWC's Task A accuracy, at <=1pp Task B cost | KG1 fired literally (0.86pp gap) but no lambda met the 1pp budget (cheapest cost 2.70pp); full sweep shows flat L2 can match EWC's number only by sacrificing far more Task B accuracy |
| 007 | [LLM-as-a-judge](007-llm-as-judge/) | Spearman rho vs human helpfulness >= 0.5; mean repeat std <= 0.75 (diagnostic: std vs error correlation) | KG1 fired (rho=0.395); KG2 passed (mean std=0.386) - precise but not accurate, confirmed by a confidently-wrong example (human=4, judge=0 all 5 repeats) |
| 008 | [LoRA on a 16 GB card](008-lora-16gb-ceiling/) | 14B without checkpointing fits without OOM; checkpointing moves the ceiling up >=1 tier | KG1 fired literally (no exception raised) but the gate measured the wrong thing - VRAM exceeded the physical 16,303 MiB by 5.9 GB via silent system-memory fallback, at a 50x step-time cost; KG2 did not fire (checkpointing left 14B still over the limit and slower, not faster) - honest ceiling is 7B |
| 009 | [Planted-PII canary](009-planted-pii-canary/) | per-type recall spread >= 20pp; targeted recall drop >= 30pp / unrelated drift < 5pp when a recognizer is disabled | KG1 passed (30.0pp en, 100.0pp bg); KG2 passed (phone recall dropped 75.0pp, nothing else moved) - and surfaced a credit-card recognizer never registered for Bulgarian at all (0.000 recall, no error) |
| 010 | [How much does an adapter forget?](010-adapter-forgetting/) | retention drop >= 5pp between smallest and largest rank/steps cell; domain gain >10pp at retention >95% for at least one cell | KG1 did not fire - retention rose 9.1pp instead of dropping, replicated 3/3 on reseed; KG2 passed in 8 of 9 cells - no forgetting observed in the tested range, and stronger fine-tuning mildly helped an unrelated general benchmark |
