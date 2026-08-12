# 003 - Blackwell quantization on my own card

Registered 2026-08-12, executed 2026-08-12. Frozen plan: [plan.md](plan.md).

**Question.** arXiv 2601.09527 reports 1.6x throughput for NVFP4 quantization
over BF16 on consumer Blackwell GPUs (RTX 5060 Ti/5070 Ti/5090), with 2-4%
quality loss. Does the same shape of trade-off hold for GGUF quantization
(what most people actually run via Ollama/llama.cpp, not NVFP4 via
TensorRT-LLM) on my own RTX 5070 Ti?

**Model and hardware.** llama3.2:3b-instruct, three quant tags (fp16, q8_0,
q4_K_M) from the official Ollama library. RTX 5070 Ti, 16 GB VRAM, driver
591.86, Windows 11, Ollama 0.32.9.

**Data.** 20 frozen prompts ([run/prompts.json](run/prompts.json)), mixed
short factual and short explanatory questions, generated at temperature 0
with a fixed seed for reproducibility.

**Registered gates and verdicts.**
- KG1: Q4_K_M throughput >= 1.3x fp16 throughput. PASSED: 258.0 tok/s vs
  116.5 tok/s = 2.22x.
- KG2: average cosine similarity (nomic-embed-text embeddings) of Q4_K_M
  output vs fp16 output, across all 20 prompts, >= 0.90. PASSED: 0.9835.
- KG3 (diagnostic, no threshold): q8_0 throughput 187.0 tok/s (1.61x),
  average similarity vs fp16 0.9984.

**VRAM delta at load:** fp16 6,895 MiB, q8_0 4,019 MiB, q4_K_M 2,594 MiB -
monotonic across all three quant levels, alongside the monotonic throughput
gain.

**Notable finding.** The two lowest-similarity prompts for q4_K_M (a
two-sentence Romeo and Juliet summary, a one-sentence firewall definition)
are open-ended explanatory questions, not closed factual ones. Reading the
raw text shows the divergence is paraphrase, not factual drift: both models
state the same facts in different words. Closed factual prompts (capital of
France, square root, chemical symbol) scored a perfect 1.0000 at every quant
level - no room for paraphrase means no measured variance. The embedding
similarity metric is correctly picking up legitimate rephrasing, not
penalising accuracy.

**Not tested here:** NVFP4 itself (different quantization scheme, different
software stack - this is context, not a replication); whether a larger
model fits at fp16 on this card (that question belongs to a separate piece).

Full data: [metrics.json](metrics.json), [throughput_vram.json](throughput_vram.json),
raw model outputs in [run/](run/).
