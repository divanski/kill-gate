"""Experiment 008: LoRA on a 16 GB card - what fits and what does not.

For each Qwen2.5-Instruct size (1.5B, 3B, 7B, 14B), loads a 4-bit quantized
checkpoint, wraps it with a LoRA adapter, and attempts one real training
step (forward + backward + optimizer.step()) at seq_len=512, batch_size=1,
both with and without gradient checkpointing. Records success/failure and
peak VRAM for each of the 8 configurations.
"""

import gc
import json
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SIZES = ["1.5B", "3B", "7B", "14B"]

# Prefer unsloth's pre-quantized bnb-4bit checkpoints (much smaller download);
# fall back to the official Qwen checkpoint with on-the-fly 4-bit quantization.
UNSLOTH_REPOS = {
    "1.5B": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    "3B": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    "7B": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "14B": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
}
OFFICIAL_REPOS = {
    "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "3B": "Qwen/Qwen2.5-3B-Instruct",
    "7B": "Qwen/Qwen2.5-7B-Instruct",
    "14B": "Qwen/Qwen2.5-14B-Instruct",
}

SEQ_LEN = 512
BATCH_SIZE = 1
LORA_RANK = 16
LORA_ALPHA = 32


def load_model(size):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    try:
        repo = UNSLOTH_REPOS[size]
        print(f"  trying pre-quantized checkpoint: {repo}")
        model = AutoModelForCausalLM.from_pretrained(
            repo, quantization_config=bnb_config, device_map={"": 0}
        )
        return model, repo
    except Exception as e:
        print(f"  unsloth checkpoint failed ({e}), falling back to official repo with on-the-fly quant")
        repo = OFFICIAL_REPOS[size]
        model = AutoModelForCausalLM.from_pretrained(
            repo, quantization_config=bnb_config, device_map={"": 0}
        )
        return model, repo


def run_one_config(size, use_checkpointing):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    result = {
        "size": size,
        "gradient_checkpointing": use_checkpointing,
        "success": False,
        "peak_vram_mib": None,
        "seconds_per_step": None,
        "error": None,
        "repo_used": None,
    }
    model = None
    try:
        model, repo = load_model(size)
        result["repo_used"] = repo

        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        if use_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

        model.train()

        torch.manual_seed(42)
        input_ids = torch.randint(0, model.config.vocab_size, (BATCH_SIZE, SEQ_LEN)).to("cuda")
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-4
        )

        start = time.time()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        elapsed = time.time() - start

        peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        result["success"] = True
        result["peak_vram_mib"] = round(peak, 1)
        result["seconds_per_step"] = round(elapsed, 3)
        print(f"  OK  size={size} gc={use_checkpointing} peak_vram={peak:.0f} MiB  {elapsed:.2f}s/step")

    except torch.cuda.OutOfMemoryError as e:
        result["error"] = f"OOM: {e}"
        print(f"  OOM size={size} gc={use_checkpointing}")
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        print(f"  ERROR size={size} gc={use_checkpointing}: {result['error']}")
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return result


def main():
    results = []
    for size in SIZES:
        for use_checkpointing in (False, True):
            print(f"\n=== size={size} gradient_checkpointing={use_checkpointing} ===")
            r = run_one_config(size, use_checkpointing)
            results.append(r)
            with open("results_partial.json", "w") as f:
                json.dump(results, f, indent=2)

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDone. Results saved to results.json")


if __name__ == "__main__":
    main()
