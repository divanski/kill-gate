"""Diagnostic: inspect raw generated text on the general_eval (MMLU) set for
the baseline model vs one trained cell, to check whether the >100% retention
seen in results_grid.json is a real effect or a parsing artifact (baseline
answering verbosely / not following the "one letter" instruction as well as
the fine-tuned model).
"""

import importlib
import json
import os
import re

import torch

g = importlib.import_module("02_run_grid")

ROOT = os.path.dirname(os.path.abspath(__file__))


def eval_general_verbose(model, tokenizer, rows):
    letters = ["A", "B", "C", "D"]
    records = []
    for row in rows:
        choices_text = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(row["choices"]))
        prompt = (
            f'Question: {row["question"]}\n{choices_text}\n\n'
            "Answer with exactly one letter: A, B, C or D."
        )
        reply = g.ask(model, tokenizer, prompt)
        match = re.search(r"[ABCD]", reply.upper())
        predicted = letters.index(match.group()) if match else None
        records.append(
            {
                "subject": row["subject"],
                "raw_reply": reply,
                "predicted": predicted,
                "answer": row["answer"],
                "correct": predicted == row["answer"],
                "unparseable": predicted is None,
            }
        )
    return records


def summarize(records, label):
    n = len(records)
    correct = sum(r["correct"] for r in records)
    unparseable = sum(r["unparseable"] for r in records)
    print(f"\n--- {label} ---")
    print(f"accuracy: {correct}/{n} = {correct/n:.4f}")
    print(f"unparseable replies: {unparseable}/{n}")
    print("sample raw replies (first 8):")
    for r in records[:8]:
        print(f"  reply={r['raw_reply']!r} predicted={r['predicted']} answer={r['answer']} correct={r['correct']}")


def main():
    general = g.load_jsonl(os.path.join(ROOT, "data", "general_eval.jsonl"))

    model, tokenizer = g.load_base_model()
    baseline_records = eval_general_verbose(model, tokenizer, general)
    summarize(baseline_records, "baseline (no adapter)")
    del model
    torch.cuda.empty_cache()

    cell_base, tokenizer = g.load_base_model()
    adapted = g.train_lora_cell(cell_base, tokenizer, g.load_jsonl(os.path.join(ROOT, "data", "overruling_train.jsonl")), rank=64, steps=500)
    adapted_records = eval_general_verbose(adapted, tokenizer, general)
    summarize(adapted_records, "rank=64 steps=500 (largest-gain cell)")

    with open(os.path.join(ROOT, "diagnostic_parsing.json"), "w", encoding="utf-8") as f:
        json.dump({"baseline": baseline_records, "rank64_steps500": adapted_records}, f, indent=2)
    print("\nSaved diagnostic_parsing.json")


if __name__ == "__main__":
    main()
