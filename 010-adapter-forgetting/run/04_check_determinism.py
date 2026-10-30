"""Sanity check requested after an odd result: reload the UNMODIFIED base
model twice, independently, and compare general_eval answers between the two
loads. If a fresh reload without any training changes the answers, the
grid's "retention" numbers are contaminated by load-to-load noise, not by
anything LoRA training did.
"""

import importlib
import json
import os

import torch

g = importlib.import_module("02_run_grid")

ROOT = os.path.dirname(os.path.abspath(__file__))


def run_once(general):
    model, tokenizer = g.load_base_model()
    letters = ["A", "B", "C", "D"]
    replies = []
    for row in general:
        choices_text = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(row["choices"]))
        prompt = (
            f'Question: {row["question"]}\n{choices_text}\n\n'
            "Answer with exactly one letter: A, B, C or D."
        )
        reply = g.ask(model, tokenizer, prompt)
        replies.append(reply)
    del model
    torch.cuda.empty_cache()
    return replies


def main():
    general = g.load_jsonl(os.path.join(ROOT, "data", "general_eval.jsonl"))

    print("Run 1 (fresh load)...")
    replies_1 = run_once(general)
    print("Run 2 (fresh load, same unmodified base model, same questions)...")
    replies_2 = run_once(general)

    diffs = [(i, a, b) for i, (a, b) in enumerate(zip(replies_1, replies_2)) if a != b]
    print(f"\nTotal questions: {len(general)}")
    print(f"Differing replies between the two identical loads: {len(diffs)}")
    for i, a, b in diffs[:20]:
        print(f"  q{i}: run1={a!r} run2={b!r}")

    with open(os.path.join(ROOT, "determinism_check.json"), "w", encoding="utf-8") as f:
        json.dump({"replies_1": replies_1, "replies_2": replies_2, "num_diffs": len(diffs)}, f, indent=2)
    print("\nSaved determinism_check.json")


if __name__ == "__main__":
    main()
