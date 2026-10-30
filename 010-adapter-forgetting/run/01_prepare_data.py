"""Prepare the Overruling (domain) and MMLU-subset (general retention) splits
for experiment 010. Writes data/overruling_{train,test}.jsonl and
data/general_eval.jsonl. Public data only.

Usage: python 01_prepare_data.py
"""

import json
import os
import random

from datasets import load_dataset

SEED = 1010
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def prepare_overruling():
    # nguha/legalbench, "overruling" config: fields are index, answer (Yes/No), text.
    # The dataset's own "train" split is a 6-example few-shot prompt set, not meant
    # for fine-tuning; we use the 2394-example "test" split and carve our own
    # train/test out of it.
    ds = load_dataset("nguha/legalbench", "overruling")
    pool = ds["test"]
    split = pool.train_test_split(test_size=0.2, seed=SEED)
    train, test = split["train"], split["test"]

    def to_jsonl(dataset, path):
        with open(path, "w", encoding="utf-8") as f:
            for row in dataset:
                label = 1 if row["answer"].strip().lower() == "yes" else 0
                f.write(json.dumps({"text": row["text"], "label": label}) + "\n")

    to_jsonl(train, os.path.join(DATA_DIR, "overruling_train.jsonl"))
    to_jsonl(test, os.path.join(DATA_DIR, "overruling_test.jsonl"))
    print(f"overruling: {len(train)} train, {len(test)} test")


def prepare_general_eval(n=120):
    ds = load_dataset("cais/mmlu", "all", split="test")
    rng = random.Random(SEED)
    idx = rng.sample(range(len(ds)), n)
    rows = ds.select(idx)
    with open(os.path.join(DATA_DIR, "general_eval.jsonl"), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    {
                        "question": row["question"],
                        "choices": row["choices"],
                        "answer": row["answer"],
                        "subject": row["subject"],
                    }
                )
                + "\n"
            )
    print(f"general_eval: {n} MMLU questions across mixed subjects")


if __name__ == "__main__":
    prepare_overruling()
    prepare_general_eval()
