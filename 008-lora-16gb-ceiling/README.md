# 008 - LoRA on a 16 GB card: what fits and what does not

Registered 2026-10-26, executed 2026-10-26/27. Frozen plan: [plan.md](plan.md).

**Question.** What is the real ceiling of model size for QLoRA-style fine-tuning
(4-bit base model + LoRA adapters) on a single 16 GB consumer GPU, and does
gradient checkpointing actually move that ceiling?

**Setup.** Qwen2.5-Instruct, four sizes (1.5B, 3B, 7B, 14B), pre-quantized
`bnb-4bit` checkpoints from unsloth. LoRA rank 16, alpha 32, on the standard
attention projections. One real training step (forward + backward +
optimizer.step()) at seq_len=512, batch_size=1, each size tested with and
without gradient checkpointing. RTX 5070 Ti, 16 GB VRAM (16,303 MiB physical).

**Registered gates and verdicts.**
- KG1: 14B without checkpointing completes without a raised OOM exception.
  FIRED by the literal wording - no exception was raised. But
  `torch.cuda.max_memory_allocated()` reported 22,189.8 MiB, 5,887 MiB **above**
  the card's physical 16,303 MiB, and the step took 21.7s instead of the
  expected sub-second range. The literal gate measured the wrong thing: Windows'
  driver-level system-memory fallback lets CUDA calls succeed past the
  physical VRAM limit instead of raising an error, at a 50x+ speed cost. Read
  correctly, 14B does **not** fit without checkpointing, consistent with the
  original hypothesis - the gate's wording, not the hypothesis, was wrong.
- KG2: gradient checkpointing moves the ceiling up by at least one tested
  tier. DID NOT FIRE - H2 rejected. With checkpointing, 14B still reported
  17,470.9 MiB (1,168 MiB over the physical limit) and took *longer* per step
  (34.5s vs 21.7s without checkpointing) - the extra recompute cost compounds
  with the paging penalty instead of relieving it.

**The number that isn't obvious from the gates alone.** Nothing in the code
or the console flagged a problem for 14B. No exception, no warning - just a
silent 50-80x slowdown that only shows up if you compare the reported VRAM
figure against the card's physical limit and notice the step-time cliff.
"Did not throw an out-of-memory error" turned out to be a bad proxy for
"fits in VRAM" on this driver/OS combination.

**Full numbers (peak VRAM / % of 16,303 MiB physical / seconds per step):**

| Size | No checkpointing | With checkpointing |
|---|---|---|
| 1.5B | 4,000.7 MiB (24.5%) / 0.79s* | 2,855.5 MiB (17.5%) / 0.25s* |
| 3B | 6,749.0 MiB (41.4%) / 0.22s | 4,490.8 MiB (27.5%) / 0.28s |
| 7B | 11,897.9 MiB (73.0%) / 0.43s | 9,709.9 MiB (59.6%) / 0.43s |
| 14B | 22,189.8 MiB (136.1%) / 21.69s | 17,470.9 MiB (107.2%) / 34.54s |

\* First two runs in the script; likely include one-time CUDA/cuDNN
initialization cost - see limitations.

The honest ceiling for this card, this configuration: **7B**, comfortably,
with or without checkpointing. Checkpointing is close to free at 7B and
below (same or less time, meaningfully less memory). At 14B it does not
rescue the setup - it just makes an already-bad situation slower.

**What it does not show.** No direct `nvidia-smi` instrumentation during the
14B runs confirming the VRAM/system-RAM split - the conclusion rests on the
allocator figure exceeding the physical limit plus the step-time cliff, not
a direct observation of the fallback mechanism itself. Only rank=16,
batch_size=1, seq_len=512 were tested; a larger rank, longer context or
larger batch would move the real ceiling down, not up. One training step,
not a full run - says nothing about memory behaviour over thousands of
steps.

Full data: [metrics.json](metrics.json), experiment code in [run/](run/).
