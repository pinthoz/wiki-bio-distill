"""Splits the data into training, validation, and test sets.

Usage: python split.py --val 150
"""

import argparse
import json
import random

from common import SYSTEM_STUDENT, messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=int, default=150)
    args = ap.parse_args()

    texts = {
        r["id"]: r["text"]
        for r in map(json.loads, open("data/raw.jsonl", encoding="utf-8"))
    }
    labels = [
        l
        for l in map(json.loads, open("data/labels.jsonl", encoding="utf-8"))
        if "label" in l
    ]
    gold = list(map(json.loads, open("data/gold.jsonl", encoding="utf-8")))
    gold_ids = {g["id"] for g in gold}  # Never train on test data
    not_bio = {g["id"] for g in gold if g["status"] == "not_bio"}

    rest = [l for l in labels if l["id"] not in gold_ids]
    random.Random(42).shuffle(rest)
    val, train = rest[: args.val], rest[args.val :]

    def to_example(lab):
        # "Conversational prompt-completion" format: loss is only counted on the response
        return {
            "id": lab["id"],
            "prompt": messages(texts[lab["id"]], SYSTEM_STUDENT),
            "completion": [
                {
                    "role": "assistant",
                    "content": json.dumps(lab["label"], ensure_ascii=False),
                }
            ],
        }

    for name, rows in (("train", train), ("val", val)):
        with open(f"data/{name}.jsonl", "w", encoding="utf-8") as f:
            for lab in rows:
                f.write(json.dumps(to_example(lab), ensure_ascii=False) + "\n")

    with open("data/test.jsonl", "w", encoding="utf-8") as f:
        for g in gold:
            if g["id"] not in not_bio:
                f.write(
                    json.dumps(
                        {
                            "id": g["id"],
                            "text": texts[g["id"]],
                            "gold": g["gold"],
                            "teacher": g["teacher"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(
        f"train={len(train)}  val={len(val)}  test={len(gold) - len(not_bio)}"
    )


if __name__ == "__main__":
    main()
