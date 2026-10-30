"""Replication check: the grid trained each (rank, steps) cell exactly once,
with no fixed seed for the LoRA adapter's random init. Re-run the smallest
(rank=4, steps=50) and largest (rank=64, steps=500) cells, each under two
different explicit seeds, to see whether the retention effect reported in
results_grid.json replicates or is run-to-run training noise.
"""

import importlib
import json
import os

import torch

g = importlib.import_module("02_run_grid")

ROOT = os.path.dirname(os.path.abspath(__file__))
SEEDS = [2026, 4242]
CELLS = [(4, 50), (64, 500)]


def run_cell(rank, steps, seed, train_rows, test_rows, general_rows):
    torch.manual_seed(seed)
    cell_base, tokenizer = g.load_base_model()
    adapted = g.train_lora_cell(cell_base, tokenizer, train_rows, rank, steps)
    domain_acc = g.eval_overruling(adapted, tokenizer, test_rows)
    general_acc = g.eval_general(adapted, tokenizer, general_rows)
    del adapted, cell_base
    torch.cuda.empty_cache()
    return domain_acc, general_acc


def main():
    train = g.load_jsonl(os.path.join(ROOT, "data", "overruling_train.jsonl"))
    test_full = g.load_jsonl(os.path.join(ROOT, "data", "overruling_test.jsonl"))
    test = test_full[: g.NUM_DOMAIN_EVAL]
    general = g.load_jsonl(os.path.join(ROOT, "data", "general_eval.jsonl"))

    baseline_model, tokenizer = g.load_base_model()
    baseline_domain = g.eval_overruling(baseline_model, tokenizer, test)
    baseline_general = g.eval_general(baseline_model, tokenizer, general)
    del baseline_model
    torch.cuda.empty_cache()
    print(f"baseline domain={baseline_domain:.4f} general={baseline_general:.4f}")

    results = {"baseline": {"domain": baseline_domain, "general": baseline_general}, "runs": []}

    for rank, steps in CELLS:
        for seed in SEEDS:
            print(f"\n=== rank={rank} steps={steps} seed={seed} ===")
            domain_acc, general_acc = run_cell(rank, steps, seed, train, test, general)
            retention = (general_acc / baseline_general) * 100
            gain = (domain_acc - baseline_domain) * 100
            print(f"domain={domain_acc:.4f} (gain {gain:+.1f}pp) general={general_acc:.4f} (retention {retention:.1f}%)")
            results["runs"].append(
                {"rank": rank, "steps": steps, "seed": seed, "domain_accuracy": domain_acc,
                 "general_accuracy": general_acc, "domain_gain_pp": gain, "retention_pct": retention}
            )

    with open(os.path.join(ROOT, "seed_variance_check.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved seed_variance_check.json")


if __name__ == "__main__":
    main()
