# Evaluate the production anonymizer stack on the planted-PII set (exp 009).
# Two passes: baseline (full stack), then with the phone pattern recognizer
# removed to simulate an accidental regression (H2).
# NOTE: imports an "anonymizer" package assembled from the standard components
# listed in plan.md (Presidio + XLM-RoBERTa NER + spaCy + lingua). That package
# is not included here - point ANONYMIZER_SRC at your own assembly of the same
# standard, open components to reproduce this.
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

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "results_canary.json")

NATIONAL_ID_GOLD = ["IDCARDNUM", "SOCIALNUM", "TAXNUM"]
GOLD_MAP = {
    "PERSON": ["GIVENNAME", "SURNAME"],
    "LOCATION": ["CITY", "STREET"],
    "EMAIL_ADDRESS": ["EMAIL"],
    "PHONE_NUMBER": ["TELEPHONENUM"],
    "CREDIT_CARD": ["CREDITCARDNUMBER"],
    "DATE_TIME": ["DATE"],
    "BG_EGN": NATIONAL_ID_GOLD,
}
GOLD_LABELS = [
    "GIVENNAME", "SURNAME", "EMAIL", "TELEPHONENUM", "CREDITCARDNUMBER",
    "DATE", "CITY", "STREET", "IDCARDNUM",
]


def final_spans(text: str):
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


def run_pass(label: str) -> dict:
    report = {}
    for lang in ("en", "bg"):
        rows = [json.loads(l) for l in open(os.path.join(DATA, f"sample_{lang}.jsonl"), encoding="utf-8")]
        recall_stats = {g: {"gold": 0, "recalled": 0} for g in GOLD_LABELS}
        t0 = time.time()
        for n, ex in enumerate(rows, 1):
            text = ex["source_text"]
            gold = [(m["label"], m["start"], m["end"]) for m in ex["privacy_mask"]]
            spans, _detected = final_spans(text)
            for gl, gs, ge in gold:
                recall_stats[gl]["gold"] += 1
                hits = [
                    s for s in spans
                    if gl in GOLD_MAP.get(s.entity_type, ()) and overlaps(s.start, s.end, gs, ge)
                ]
                if hits:
                    recall_stats[gl]["recalled"] += 1
            if n % 80 == 0:
                print(f"[{label}/{lang}] {n}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
        report[lang] = {"examples": len(rows), "seconds": round(time.time() - t0, 1), "recall": recall_stats}
    return report


def recall_table(report: dict) -> dict:
    table = {}
    for lang, r in report.items():
        table[lang] = {
            g: round(s["recalled"] / s["gold"], 4)
            for g, s in r["recall"].items() if s["gold"]
        }
    return table


def main() -> None:
    print("warmup (baseline stack)...", flush=True)
    t0 = time.time()
    engine.warmup()
    print(f"warmup done in {time.time()-t0:.0f}s", flush=True)

    baseline = run_pass("baseline")

    print("\ndisabling the phone pattern recognizer for the H2 regression pass...", flush=True)
    analyzer = engine._build_analyzer()
    analyzer.registry.remove_recognizer("BoostedPhoneRecognizer")
    degraded = run_pass("phone-disabled")

    out = {"baseline": baseline, "phone_disabled": degraded}
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2)

    base_table = recall_table(baseline)
    deg_table = recall_table(degraded)

    print("\n=== Recall by type (baseline) ===")
    for lang in ("en", "bg"):
        print(f"-- {lang} --")
        for g, v in base_table[lang].items():
            print(f"  {g:<16} {v:.3f}")

    print("\n=== KG1: spread between best and worst type ===")
    for lang in ("en", "bg"):
        vals = base_table[lang]
        best_label = max(vals, key=vals.get)
        worst_label = min(vals, key=vals.get)
        spread = (vals[best_label] - vals[worst_label]) * 100
        print(f"  {lang}: best={best_label} ({vals[best_label]:.3f}) worst={worst_label} ({vals[worst_label]:.3f}) spread={spread:.1f}pp"
              f" -> KG1 {'PASS (H1 holds)' if spread >= 20 else 'FAIL (H1 rejected)'}")

    print("\n=== KG2: delta recall after disabling phone recognizer ===")
    for lang in ("en", "bg"):
        print(f"-- {lang} --")
        targeted_delta = None
        unrelated_max_delta = 0.0
        for g in base_table[lang]:
            delta = (base_table[lang][g] - deg_table[lang].get(g, 0.0)) * 100
            tag = "TELEPHONENUM (targeted)" if g == "TELEPHONENUM" else "unrelated"
            print(f"  {g:<16} baseline={base_table[lang][g]:.3f} degraded={deg_table[lang].get(g, 0.0):.3f} delta={delta:+.1f}pp ({tag})")
            if g == "TELEPHONENUM":
                targeted_delta = delta
            else:
                unrelated_max_delta = max(unrelated_max_delta, abs(delta))
        kg2_pass = targeted_delta is not None and targeted_delta >= 30 and unrelated_max_delta < 5
        print(f"  -> KG2 {'PASS (H2 holds)' if kg2_pass else 'FAIL (H2 rejected)'}"
              f" (targeted_delta={targeted_delta:.1f}pp, max_unrelated_delta={unrelated_max_delta:.1f}pp)")


if __name__ == "__main__":
    main()
