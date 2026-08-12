# Run all 20 frozen prompts through each quant variant, record throughput
# (from Ollama's own eval_count/eval_duration), VRAM (ollama ps + nvidia-smi),
# and raw output text for later embedding-similarity scoring.
import io
import json
import os
import subprocess
import sys
import time
import urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = json.load(open(os.path.join(HERE, "prompts.json"), encoding="utf-8"))["prompts"]
TAGS = ["fp16", "q8_0", "q4_K_M"]
MODEL_BASE = "llama3.2:3b-instruct"
OLLAMA = "http://localhost:11434"


def ollama_stop_all():
    subprocess.run(["ollama", "stop", "all"], capture_output=True, text=True)
    time.sleep(1)


def generate(model, prompt):
    req = urllib.request.Request(
        f"{OLLAMA}/api/generate",
        data=json.dumps({
            "model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0, "seed": 42},
        }).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def ollama_ps():
    out = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
    return out


def nvidia_vram_used_mib():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.strip()
    return int(out.splitlines()[0])


def main():
    results = {}
    for tag in TAGS:
        model = f"{MODEL_BASE}-{tag}"
        print(f"\n=== {model} ===", flush=True)
        ollama_stop_all()
        vram_before = nvidia_vram_used_mib()

        # warmup / load
        generate(model, "Hello.")
        vram_loaded = nvidia_vram_used_mib()
        ps_output = ollama_ps()

        outputs = []
        t0 = time.time()
        for p in PROMPTS:
            r = generate(model, p["text"])
            eval_count = r.get("eval_count", 0)
            eval_ns = r.get("eval_duration", 1)
            tok_per_s = eval_count / (eval_ns / 1e9) if eval_ns else None
            outputs.append({
                "id": p["id"], "prompt": p["text"], "response": r.get("response", ""),
                "eval_count": eval_count, "eval_duration_ns": eval_ns,
                "tokens_per_sec": tok_per_s,
                "load_duration_ns": r.get("load_duration", 0),
                "total_duration_ns": r.get("total_duration", 0),
            })
            print(f"  [{p['id']:2d}/20] {tok_per_s:.1f} tok/s" if tok_per_s else f"  [{p['id']:2d}/20] n/a", flush=True)
        elapsed = time.time() - t0

        json.dump({"model": model, "outputs": outputs}, open(os.path.join(HERE, f"outputs_{tag}.json"), "w", encoding="utf-8"), indent=2)

        toks = [o["tokens_per_sec"] for o in outputs if o["tokens_per_sec"]]
        results[tag] = {
            "model": model,
            "vram_before_mib": vram_before,
            "vram_loaded_mib": vram_loaded,
            "vram_delta_mib": vram_loaded - vram_before,
            "ollama_ps_at_load": ps_output.strip(),
            "avg_tokens_per_sec": sum(toks) / len(toks) if toks else None,
            "min_tokens_per_sec": min(toks) if toks else None,
            "max_tokens_per_sec": max(toks) if toks else None,
            "wall_clock_seconds_for_20_prompts": elapsed,
        }
        print(f"  avg {results[tag]['avg_tokens_per_sec']:.1f} tok/s | VRAM delta {results[tag]['vram_delta_mib']} MiB", flush=True)

    ollama_stop_all()
    json.dump(results, open(os.path.join(HERE, "throughput_vram.json"), "w", encoding="utf-8"), indent=2)
    print("\nDONE. Wrote throughput_vram.json and outputs_<tag>.json")


if __name__ == "__main__":
    main()
