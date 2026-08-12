# Embed every output with nomic-embed-text, compute cosine similarity of
# q8_0 and q4_K_M against fp16 per prompt, then apply the registered
# kill-gates (KG1: throughput, KG2: similarity) and write the final verdict.
import io
import json
import math
import os
import sys
import urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
OLLAMA = "http://localhost:11434"
KG1_THROUGHPUT_MULT = 1.3
KG2_SIMILARITY_MIN = 0.90


def embed(text):
    req = urllib.request.Request(
        f"{OLLAMA}/api/embeddings",
        data=json.dumps({"model": "nomic-embed-text", "prompt": text}).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["embedding"]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def main():
    outs = {}
    for tag in ["fp16", "q8_0", "q4_K_M"]:
        outs[tag] = {o["id"]: o["response"] for o in json.load(open(os.path.join(HERE, f"outputs_{tag}.json"), encoding="utf-8"))["outputs"]}

    sims = {"q8_0": [], "q4_K_M": []}
    per_prompt = []
    for pid in sorted(outs["fp16"]):
        ref = outs["fp16"][pid]
        ref_emb = embed(ref)
        row = {"id": pid}
        for tag in ["q8_0", "q4_K_M"]:
            e = embed(outs[tag][pid])
            s = cosine(ref_emb, e)
            sims[tag].append(s)
            row[f"similarity_{tag}_vs_fp16"] = s
        per_prompt.append(row)
        print(f"  [{pid:2d}/20] q8_0={row['similarity_q8_0_vs_fp16']:.4f}  q4_K_M={row['similarity_q4_K_M_vs_fp16']:.4f}", flush=True)

    avg_sim_q8 = sum(sims["q8_0"]) / len(sims["q8_0"])
    avg_sim_q4 = sum(sims["q4_K_M"]) / len(sims["q4_K_M"])

    throughput = json.load(open(os.path.join(HERE, "throughput_vram.json"), encoding="utf-8"))
    fp16_tps = throughput["fp16"]["avg_tokens_per_sec"]
    q8_tps = throughput["q8_0"]["avg_tokens_per_sec"]
    q4_tps = throughput["q4_K_M"]["avg_tokens_per_sec"]

    kg1_ratio = q4_tps / fp16_tps if fp16_tps else None
    kg1_pass = kg1_ratio is not None and kg1_ratio >= KG1_THROUGHPUT_MULT
    kg2_pass = avg_sim_q4 >= KG2_SIMILARITY_MIN

    verdict = {
        "KG1_throughput": {
            "threshold": f">= {KG1_THROUGHPUT_MULT}x fp16",
            "q4_K_M_tokens_per_sec": q4_tps,
            "fp16_tokens_per_sec": fp16_tps,
            "ratio": kg1_ratio,
            "verdict": "PASSED" if kg1_pass else "FIRED (H1 rejected)",
        },
        "KG2_quality": {
            "threshold": f">= {KG2_SIMILARITY_MIN} avg cosine similarity",
            "avg_similarity_q4_K_M_vs_fp16": avg_sim_q4,
            "verdict": "PASSED" if kg2_pass else "FIRED (H2 rejected)",
        },
        "KG3_diagnostic_q8_0": {
            "tokens_per_sec": q8_tps,
            "ratio_vs_fp16": q8_tps / fp16_tps if fp16_tps else None,
            "avg_similarity_vs_fp16": avg_sim_q8,
            "note": "diagnostic only, no pass/fail threshold registered",
        },
    }

    out = {
        "per_prompt_similarity": per_prompt,
        "avg_similarity": {"q8_0_vs_fp16": avg_sim_q8, "q4_K_M_vs_fp16": avg_sim_q4},
        "throughput_vram": throughput,
        "verdict": verdict,
    }
    json.dump(out, open(os.path.join(HERE, "metrics.json"), "w", encoding="utf-8"), indent=2)

    print("\n=== SUMMARY ===")
    print(f"fp16   : {fp16_tps:.1f} tok/s | VRAM delta {throughput['fp16']['vram_delta_mib']} MiB")
    print(f"q8_0   : {q8_tps:.1f} tok/s | VRAM delta {throughput['q8_0']['vram_delta_mib']} MiB | sim vs fp16 {avg_sim_q8:.4f}")
    print(f"q4_K_M : {q4_tps:.1f} tok/s | VRAM delta {throughput['q4_K_M']['vram_delta_mib']} MiB | sim vs fp16 {avg_sim_q4:.4f}")
    print(f"\nKG1 (throughput >= {KG1_THROUGHPUT_MULT}x): ratio={kg1_ratio:.3f} -> {verdict['KG1_throughput']['verdict']}")
    print(f"KG2 (similarity >= {KG2_SIMILARITY_MIN}): avg={avg_sim_q4:.4f} -> {verdict['KG2_quality']['verdict']}")


if __name__ == "__main__":
    main()
