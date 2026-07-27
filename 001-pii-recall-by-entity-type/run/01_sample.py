# Freeze the evaluation sample: first 2000 examples per language (en, bg)
# in native order from ai4privacy/pii-masking-openpii-1m data/train.jsonl.
import io
import json
import os
import sys

from datasets import load_dataset

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TARGET_PER_LANG = 2000
LANGS = ("en", "bg")
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)

counts = {lang: 0 for lang in LANGS}
writers = {
    lang: open(os.path.join(OUT_DIR, f"sample_{lang}.jsonl"), "w", encoding="utf-8", newline="")
    for lang in LANGS
}

ds = load_dataset("ai4privacy/pii-masking-openpii-1m", split="train", streaming=True)
scanned = 0
for ex in ds:
    scanned += 1
    lang = ex["language"]
    if lang in counts and counts[lang] < TARGET_PER_LANG:
        row = {
            "uid": ex["uid"],
            "language": lang,
            "script": ex["script"],
            "source_text": ex["source_text"],
            "privacy_mask": ex["privacy_mask"],
        }
        writers[lang].write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[lang] += 1
    if all(c >= TARGET_PER_LANG for c in counts.values()):
        break
    if scanned % 50000 == 0:
        print(f"scanned {scanned}, collected {counts}", flush=True)

for w in writers.values():
    w.close()
print(f"DONE: scanned {scanned} rows, collected {counts}")
