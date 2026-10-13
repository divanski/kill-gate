# LLM-as-a-judge agreement with human ratings, and the judge's own
# repeat-to-repeat stability, on nvidia/HelpSteer2 (public, CC-BY-4.0).
# Judge: gpt-oss:latest via local Ollama (RTX 5070 Ti, 16GB).
import gzip
import json
import os
import random
import time

import requests
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
SAMPLE_SIZE = 150
N_REPEATS = 5
TEMPERATURE = 0.7
JUDGE_MODEL = "gpt-oss:latest"
OLLAMA_URL = "http://localhost:11434/api/generate"

RUBRIC = """Rate the HELPFULNESS of the RESPONSE to the PROMPT on a scale of 0 to 4, where:
0 = not helpful at all
1 = slightly helpful
2 = moderately helpful
3 = helpful
4 = extremely helpful, complete and clear

PROMPT: {prompt}

RESPONSE: {response}

Reply with ONLY a single digit from 0 to 4, nothing else."""


def load_validation_set():
    path = os.path.join(HERE, "..", "data", "validation.jsonl.gz")
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def judge_once(prompt_text, response_text):
    rubric = RUBRIC.format(prompt=prompt_text, response=response_text)
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": JUDGE_MODEL,
            "prompt": rubric,
            "stream": False,
            "options": {"temperature": TEMPERATURE},
        },
        timeout=60,
    )
    resp.raise_for_status()
    text = resp.json()["response"].strip()
    for ch in text:
        if ch.isdigit() and ch in "01234":
            return int(ch)
    return None


def main():
    t0 = time.time()
    random.seed(SEED)
    records = load_validation_set()
    print(f"validation set size: {len(records)}")

    indices = list(range(len(records)))
    random.shuffle(indices)
    sample_indices = indices[:SAMPLE_SIZE]

    results = []
    for n, idx in enumerate(sample_indices):
        rec = records[idx]
        scores = []
        for _ in range(N_REPEATS):
            score = judge_once(rec["prompt"], rec["response"])
            if score is not None:
                scores.append(score)
        if not scores:
            print(f"  item {n}: no parseable judge scores, skipping")
            continue
        mean_score = sum(scores) / len(scores)
        std_score = stats.tstd(scores) if len(scores) > 1 else 0.0
        results.append({
            "index": idx,
            "human_helpfulness": rec["helpfulness"],
            "judge_scores": scores,
            "judge_mean": mean_score,
            "judge_std": std_score,
            "abs_error": abs(mean_score - rec["helpfulness"]),
        })
        if (n + 1) % 10 == 0:
            print(f"  {n + 1}/{len(sample_indices)} done, elapsed {time.time() - t0:.0f}s", flush=True)

    human_scores = [r["human_helpfulness"] for r in results]
    judge_means = [r["judge_mean"] for r in results]
    judge_stds = [r["judge_std"] for r in results]
    abs_errors = [r["abs_error"] for r in results]

    spearman_rho, spearman_p = stats.spearmanr(human_scores, judge_means)
    mean_std = sum(judge_stds) / len(judge_stds)
    std_vs_error_rho, std_vs_error_p = stats.spearmanr(judge_stds, abs_errors)

    kg1_pass = spearman_rho >= 0.5
    kg2_pass = mean_std <= 0.75

    summary = {
        "config": {
            "seed": SEED,
            "sample_size": SAMPLE_SIZE,
            "n_repeats": N_REPEATS,
            "temperature": TEMPERATURE,
            "judge_model": JUDGE_MODEL,
            "n_valid_items": len(results),
            "wall_clock_seconds": time.time() - t0,
        },
        "spearman_rho_judge_vs_human": spearman_rho,
        "spearman_p_judge_vs_human": spearman_p,
        "mean_judge_std": mean_std,
        "spearman_rho_std_vs_error": std_vs_error_rho,
        "spearman_p_std_vs_error": std_vs_error_p,
        "KG1_agreement": "PASSED" if kg1_pass else "FIRED (H1 rejected)",
        "KG2_stability": "PASSED" if kg2_pass else "FIRED (H2 rejected)",
        "items": results,
    }

    out_path = os.path.join(HERE, "..", "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"n items: {len(results)}")
    print(f"Spearman rho (judge mean vs human): {spearman_rho:.3f} (p={spearman_p:.4f})")
    print(f"Mean judge std across {N_REPEATS} repeats: {mean_std:.3f}")
    print(f"Spearman rho (judge std vs abs error): {std_vs_error_rho:.3f} (p={std_vs_error_p:.4f})")
    print(f"\nKG1 (rho >= 0.5): {summary['KG1_agreement']}")
    print(f"KG2 (mean std <= 0.75): {summary['KG2_stability']}")
    print(f"\nwall clock: {summary['config']['wall_clock_seconds']:.1f}s")


if __name__ == "__main__":
    main()
