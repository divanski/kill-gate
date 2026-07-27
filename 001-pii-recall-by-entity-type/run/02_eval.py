# Evaluate GLiNER2-PII per-entity-type on the frozen en/bg samples.
# Matching: model outputs strings (no offsets), so predicted spans are located
# as all occurrences of the predicted string in source_text.
#   - relaxed: gold span is recalled if any predicted occurrence of the same
#     mapped label overlaps it by at least one character
#   - strict:  an occurrence matches gold boundaries exactly
# Precision: a predicted string is a true positive if any of its occurrences
# overlaps a gold span of the same label.
import io
import json
import os
import sys
import time

import torch
from gliner2 import GLiNER2

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = os.path.join(os.path.dirname(__file__), "data")
OUT = os.path.join(os.path.dirname(__file__), "metrics.json")

LABEL_MAP = {
    "GIVENNAME": "given name",
    "SURNAME": "surname",
    "TITLE": "honorific title",
    "DATE": "date",
    "CITY": "city",
    "STREET": "street name",
    "BUILDINGNUM": "building number",
    "ZIPCODE": "postal code",
    "EMAIL": "email address",
    "TELEPHONENUM": "phone number",
    "AGE": "age",
    "GENDER": "gender identity",
    "SEX": "sex",
    "CREDITCARDNUMBER": "credit card number",
    "IDCARDNUM": "identity card number",
    "DRIVERLICENSENUM": "driver's license number",
    "TAXNUM": "tax identification number",
    "SOCIALNUM": "social security number",
    "PASSPORTNUM": "passport number",
}
MODEL_LABELS = list(LABEL_MAP.values())
INV = {v: k for k, v in LABEL_MAP.items()}


def occurrences(text, needle):
    spans, start = [], 0
    if not needle:
        return spans
    while True:
        i = text.find(needle, start)
        if i == -1:
            return spans
        spans.append((i, i + len(needle)))
        start = i + 1


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def main():
    model = GLiNER2.from_pretrained("fastino/gliner2-privacy-filter-PII-multi")
    device = "cpu"
    try:
        model = model.to("cuda")
        device = "cuda"
        torch.cuda.reset_peak_memory_stats()
    except Exception as e:
        print("GPU move failed, staying on CPU:", e)

    results = {}
    for lang in ("en", "bg"):
        rows = [json.loads(l) for l in open(os.path.join(BASE, f"sample_{lang}.jsonl"), encoding="utf-8")]
        stats = {k: {"gold": 0, "recalled": 0, "recalled_strict": 0, "pred": 0, "pred_tp": 0} for k in LABEL_MAP}
        t0 = time.time()
        for n, ex in enumerate(rows, 1):
            text = ex["source_text"]
            gold = [(m["label"], m["start"], m["end"]) for m in ex["privacy_mask"]]
            try:
                out = model.extract_entities(text, MODEL_LABELS)
            except Exception as err:
                print(f"[{lang}] example {n} failed: {err}")
                continue
            pred = {}  # dataset label -> list of (start, end)
            for mlabel, values in out.get("entities", {}).items():
                dlabel = INV.get(mlabel)
                if dlabel is None:
                    continue
                for val in values:
                    occ = occurrences(text, val)
                    stats[dlabel]["pred"] += 1
                    hit = any(
                        overlaps(o, (gs, ge))
                        for o in occ
                        for gl, gs, ge in gold
                        if gl == dlabel
                    )
                    if hit:
                        stats[dlabel]["pred_tp"] += 1
                    pred.setdefault(dlabel, []).extend(occ)
            for gl, gs, ge in gold:
                stats[gl]["gold"] += 1
                p = pred.get(gl, [])
                if any(overlaps(o, (gs, ge)) for o in p):
                    stats[gl]["recalled"] += 1
                if any(o == (gs, ge) for o in p):
                    stats[gl]["recalled_strict"] += 1
            if n % 250 == 0:
                print(f"[{lang}] {n}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
        results[lang] = {
            "examples": len(rows),
            "seconds": round(time.time() - t0, 1),
            "stats": stats,
        }

    meta = {"device": device, "model": "fastino/gliner2-privacy-filter-PII-multi"}
    if device == "cuda":
        meta["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        meta["gpu"] = torch.cuda.get_device_name(0)
    json.dump({"meta": meta, "results": results}, open(OUT, "w", encoding="utf-8"), indent=2)
    print("META:", meta)
    for lang in results:
        print(f"\n=== {lang} (n={results[lang]['examples']}, {results[lang]['seconds']}s) ===")
        print(f"{'label':<18} {'gold':>5} {'recall':>7} {'strict':>7} {'precision':>9}")
        for k, s in sorted(results[lang]["stats"].items()):
            if s["gold"] == 0:
                continue
            r = s["recalled"] / s["gold"]
            rs = s["recalled_strict"] / s["gold"]
            p = s["pred_tp"] / s["pred"] if s["pred"] else 0.0
            print(f"{k:<18} {s['gold']:>5} {r:>7.3f} {rs:>7.3f} {p:>9.3f}")


if __name__ == "__main__":
    main()
