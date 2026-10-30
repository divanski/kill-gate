"""Run the 3x3 LoRA rank x steps grid on Qwen2.5-3B-Instruct, evaluating
domain accuracy (Overruling) and general retention (MMLU subset) for every
cell, plus the unadapted baseline. Writes results_grid.json.

Frozen per plan.md, approved 2026-10-30: rank {4, 16, 64} x steps {50, 200, 500}.

Usage: python 02_run_grid.py
"""

import json
import os
import re

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
RANKS = [4, 16, 64]
STEP_COUNTS = [50, 200, 500]
SEED = 1010
NUM_DOMAIN_EVAL = 150  # subsample of the 479-example test split, for runtime


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_base_model():
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="cuda:0"
    )
    return model, tokenizer


def ask(model, tokenizer, prompt, max_new_tokens=4):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return reply.strip()


def eval_overruling(model, tokenizer, rows):
    correct = 0
    for row in rows:
        prompt = (
            f'Sentence: "{row["text"]}"\n\n'
            "Does this sentence overrule a prior case holding? Answer with exactly one word, yes or no."
        )
        reply = ask(model, tokenizer, prompt).lower()
        predicted = 1 if "yes" in reply else 0 if "no" in reply else None
        if predicted == row["label"]:
            correct += 1
    return correct / len(rows)


def eval_general(model, tokenizer, rows):
    letters = ["A", "B", "C", "D"]
    correct = 0
    for row in rows:
        choices_text = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(row["choices"]))
        prompt = (
            f'Question: {row["question"]}\n{choices_text}\n\n'
            "Answer with exactly one letter: A, B, C or D."
        )
        reply = ask(model, tokenizer, prompt).upper()
        match = re.search(r"[ABCD]", reply)
        predicted = letters.index(match.group()) if match else None
        if predicted == row["answer"]:
            correct += 1
    return correct / len(rows)


def train_lora_cell(base_model, tokenizer, train_rows, rank, steps):
    config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, config)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    step = 0
    while step < steps:
        for row in train_rows:
            if step >= steps:
                break
            prompt = (
                f'Sentence: "{row["text"]}"\n\n'
                "Does this sentence overrule a prior case holding? Answer with exactly one word, yes or no."
            )
            target = "yes" if row["label"] == 1 else "no"
            messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": target}]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1

    model.eval()
    return model


def main():
    overruling_train = load_jsonl(os.path.join(DATA_DIR, "overruling_train.jsonl"))
    overruling_test_full = load_jsonl(os.path.join(DATA_DIR, "overruling_test.jsonl"))
    overruling_test = overruling_test_full[:NUM_DOMAIN_EVAL]
    general_eval = load_jsonl(os.path.join(DATA_DIR, "general_eval.jsonl"))
    print(f"domain eval subsample: {len(overruling_test)} of {len(overruling_test_full)}")

    model, tokenizer = load_base_model()

    print("Evaluating baseline (no adapter)...")
    baseline_domain = eval_overruling(model, tokenizer, overruling_test)
    baseline_general = eval_general(model, tokenizer, general_eval)
    print(f"baseline domain accuracy: {baseline_domain:.4f}")
    print(f"baseline general accuracy: {baseline_general:.4f}")

    # Free the baseline model before each cell: get_peft_model() injects adapter
    # layers into the model it is given, in place. Reusing the same loaded model
    # object across cells would silently stack each cell's training on top of the
    # previous one's adapter instead of starting from the clean baseline every
    # time. Reloading from scratch per cell costs ~15s but makes each cell
    # independent, which is the whole point of the comparison.
    del model
    torch.cuda.empty_cache()

    results = {
        "model": MODEL_ID,
        "baseline": {"domain_accuracy": baseline_domain, "general_accuracy": baseline_general},
        "cells": [],
    }

    for rank in RANKS:
        for steps in STEP_COUNTS:
            print(f"\n=== rank={rank} steps={steps} ===")
            cell_base, tokenizer = load_base_model()
            adapted = train_lora_cell(cell_base, tokenizer, overruling_train, rank, steps)
            domain_acc = eval_overruling(adapted, tokenizer, overruling_test)
            general_acc = eval_general(adapted, tokenizer, general_eval)
            domain_gain_pp = (domain_acc - baseline_domain) * 100
            retention_pct = (general_acc / baseline_general) * 100 if baseline_general > 0 else None
            print(f"domain accuracy: {domain_acc:.4f} (gain {domain_gain_pp:+.1f}pp)")
            print(f"general accuracy: {general_acc:.4f} (retention {retention_pct:.1f}%)")
            results["cells"].append(
                {
                    "rank": rank,
                    "steps": steps,
                    "domain_accuracy": domain_acc,
                    "general_accuracy": general_acc,
                    "domain_gain_pp": domain_gain_pp,
                    "retention_pct": retention_pct,
                }
            )
            del adapted, cell_base
            torch.cuda.empty_cache()

    with open(os.path.join(ROOT, "..", "results_grid.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nSaved results_grid.json")


if __name__ == "__main__":
    main()
