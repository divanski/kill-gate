# Evaluate the ensemble configuration on the frozen samples of exp 001.
# NOTE: imports an "anonymizer" package assembled from the standard components
# listed in the plan; that assembly is not part of this repository.
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("ANONYMIZER_SRC", os.path.join(HERE, "anonymizer", "src")))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from anonymizer import engine  # noqa: E402
from anonymizer.config import get_settings  # noqa: E402
from anonymizer.language import detect_language  # noqa: E402
from anonymizer.recognizers.denylist import (  # noqa: E402
    filter_denylisted,
    filter_protected_brands,
)

DATA = os.path.join(HERE, "..", "..", "001-pii-recall-by-entity-type", "run", "data")
OUT = os.path.join(HERE, "metrics_stack.json")

NATIONAL_ID_GOLD = ["IDCARDNUM", "SOCIALNUM", "TAXNUM"]
GOLD_MAP = {
    "PERSON": ["GIVENNAME", "SURNAME"],
    "LOCATION": ["CITY", "STREET"],
    "EMAIL_ADDRESS": ["EMAIL"],
    "PHONE_NUMBER": ["TELEPHONENUM"],
    "CREDIT_CARD": ["CREDITCARDNUMBER"],
    "DATE_TIME": ["DATE"],
    "BG_EGN": NATIONAL_ID_GOLD,
    "RO_CNP": NATIONAL_ID_GOLD,
    "GR_AMKA": NATIONAL_ID_GOLD,
    "HR_OIB": NATIONAL_ID_GOLD,
    "RS_JMBG": NATIONAL_ID_GOLD,
    "ES_NIF": NATIONAL_ID_GOLD,
    "ES_NIE": NATIONAL_ID_GOLD,
    "IT_FISCAL_CODE": NATIONAL_ID_GOLD,
    "PL_PESEL": NATIONAL_ID_GOLD,
    "UK_NHS": NATIONAL_ID_GOLD,
}
CONFUSABLE_NUMERIC = {
    "CREDITCARDNUMBER", "IDCARDNUM", "SOCIALNUM", "TAXNUM",
    "DRIVERLICENSENUM", "PASSPORTNUM", "ZIPCODE",
}
GOLD_LABELS = [
    "GIVENNAME", "SURNAME", "TITLE", "DATE", "CITY", "STREET", "BUILDINGNUM",
    "ZIPCODE", "EMAIL", "TELEPHONENUM", "AGE", "GENDER", "SEX",
    "CREDITCARDNUMBER", "IDCARDNUM", "DRIVERLICENSENUM", "TAXNUM",
    "SOCIALNUM", "PASSPORTNUM",
]


def final_spans(text: str):
    """engine.anonymize() lines up to placeholders, returning spans + detected lang."""
    detected = detect_language(text)
    routing = engine._route_language(detected)
    spans = engine._spans_from_presidio(text, routing) + engine._spans_from_ner(text)
    spans = engine._strip_span_whitespace(text, spans)
    spans = engine._dedupe_overlaps(spans)
    spans = filter_denylisted(text, spans)
    spans = filter_protected_brands(text, spans, get_settings().protected_brands)
    return spans, detected


def overlaps(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end


def main():
    print("warmup...", flush=True)
    t0 = time.time()
    engine.warmup()
    print(f"warmup done in {time.time()-t0:.0f}s", flush=True)

    report = {}
    for lang in ("en", "bg"):
        rows = [json.loads(l) for l in open(os.path.join(DATA, f"sample_{lang}.jsonl"), encoding="utf-8")]
        recall_stats = {g: {"gold": 0, "recalled": 0, "recalled_strict": 0} for g in GOLD_LABELS}
        pred_stats = {}  # stack entity type -> {pred, tp}
        phone_axis = {"tp": 0, "dangerous_fp": 0, "other": 0}
        lang_detect = {"match": 0, "mismatch": 0, "examples": []}
        t0 = time.time()
        for n, ex in enumerate(rows, 1):
            text = ex["source_text"]
            gold = [(m["label"], m["start"], m["end"]) for m in ex["privacy_mask"]]
            spans, detected = final_spans(text)
            if detected == lang:
                lang_detect["match"] += 1
            else:
                lang_detect["mismatch"] += 1
                if len(lang_detect["examples"]) < 5:
                    lang_detect["examples"].append(detected)
            for s in spans:
                st = pred_stats.setdefault(s.entity_type, {"pred": 0, "tp": 0})
                st["pred"] += 1
                targets = GOLD_MAP.get(s.entity_type)
                if targets and any(
                    overlaps(s.start, s.end, gs, ge) for gl, gs, ge in gold if gl in targets
                ):
                    st["tp"] += 1
                if s.entity_type == "PHONE_NUMBER":
                    if any(overlaps(s.start, s.end, gs, ge) for gl, gs, ge in gold if gl == "TELEPHONENUM"):
                        phone_axis["tp"] += 1
                    elif any(
                        overlaps(s.start, s.end, gs, ge)
                        for gl, gs, ge in gold
                        if gl in CONFUSABLE_NUMERIC
                    ):
                        phone_axis["dangerous_fp"] += 1
                    else:
                        phone_axis["other"] += 1
            for gl, gs, ge in gold:
                recall_stats[gl]["gold"] += 1
                hits = [
                    s for s in spans
                    if gl in GOLD_MAP.get(s.entity_type, ()) and overlaps(s.start, s.end, gs, ge)
                ]
                if hits:
                    recall_stats[gl]["recalled"] += 1
                if any(s.start == gs and s.end == ge for s in hits):
                    recall_stats[gl]["recalled_strict"] += 1
            if n % 100 == 0:
                print(f"[{lang}] {n}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
        report[lang] = {
            "examples": len(rows),
            "seconds": round(time.time() - t0, 1),
            "recall": recall_stats,
            "precision": pred_stats,
            "phone_axis": phone_axis,
            "language_detection": lang_detect,
        }

    json.dump(report, open(OUT, "w", encoding="utf-8"), indent=2)
    for lang in report:
        r = report[lang]
        print(f"\n=== {lang} (n={r['examples']}, {r['seconds']}s) ===")
        print(f"{'gold label':<18} {'gold':>5} {'recall':>7} {'strict':>7}")
        for g, s in r["recall"].items():
            if s["gold"]:
                print(f"{g:<18} {s['gold']:>5} {s['recalled']/s['gold']:>7.3f} {s['recalled_strict']/s['gold']:>7.3f}")
        print(f"{'stack type':<18} {'pred':>5} {'precision':>9}")
        for t, s in sorted(r["precision"].items()):
            if s["pred"]:
                print(f"{t:<18} {s['pred']:>5} {s['tp']/s['pred']:>9.3f}")
        pa = r["phone_axis"]
        denom = pa["tp"] + pa["dangerous_fp"]
        kg2b = pa["tp"] / denom if denom else float("nan")
        print(f"phone axis: tp={pa['tp']} dangerous_fp={pa['dangerous_fp']} other={pa['other']} kg2b={kg2b:.4f}")
        print("language detection:", r["language_detection"]["match"], "match /",
              r["language_detection"]["mismatch"], "mismatch", r["language_detection"]["examples"])


if __name__ == "__main__":
    main()
