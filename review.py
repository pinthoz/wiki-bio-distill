"""Human review of the gold test set.

Shows each text and the teacher's label; accept with Enter or correct in the editor.
Usage: python review.py --n 150   (can be interrupted and resumed)
       python review.py --n 150 --accept-all   (accept every teacher label, no prompts)
"""

import argparse
import json
import os
import random
import subprocess
import tempfile

from pydantic import ValidationError

from common import Biography


def edit(label: dict) -> dict:
    with tempfile.NamedTemporaryFile(
        "w+", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(label, tmp, ensure_ascii=False, indent=2)
    while True:
        subprocess.run([os.environ.get("EDITOR", "nano"), tmp.name])
        try:
            with open(tmp.name, encoding="utf-8") as f:
                return Biography.model_validate(json.load(f)).model_dump()
        except (json.JSONDecodeError, ValidationError) as e:
            input(f"Invalid JSON ({str(e)[:120]}). Press Enter to go back to the editor.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--raw", default="data/raw.jsonl")
    ap.add_argument("--labels", default="data/labels.jsonl")
    ap.add_argument("--out", default="data/gold.jsonl")
    ap.add_argument(
        "--accept-all", action="store_true", help="accept all labels without prompting"
    )
    args = ap.parse_args()

    texts = {r["id"]: r for r in map(json.loads, open(args.raw, encoding="utf-8"))}
    labels = [
        l for l in map(json.loads, open(args.labels, encoding="utf-8")) if "label" in l
    ]
    random.Random(7).shuffle(labels)  # deterministic sample
    sample = labels[: args.n]

    done = {}
    if os.path.exists(args.out):
        done = {g["id"]: g for g in map(json.loads, open(args.out, encoding="utf-8"))}

    for i, lab in enumerate(sample, 1):
        if lab["id"] in done:
            continue
        if args.accept_all:
            rec = {"id": lab["id"], "teacher": lab["label"]}
            rec.update(gold=lab["label"], status="accepted")
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            continue
        row = texts[lab["id"]]
        os.system("clear")
        print(f"[{i}/{len(sample)}] {row['title']}\n\n{row['text']}\n")
        print(json.dumps(lab["label"], ensure_ascii=False, indent=2))
        ans = (
            input("\n[Enter] accept  [e] edit  [x] not a biography  [q] quit > ")
            .strip()
            .lower()
        )
        if ans == "q":
            break
        rec = {"id": lab["id"], "teacher": lab["label"]}
        if ans == "x":
            rec.update(gold=None, status="not_bio")
        elif ans == "e":
            rec.update(gold=edit(lab["label"]), status="edited")
        else:
            rec.update(gold=lab["label"], status="accepted")
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    gold = list(map(json.loads, open(args.out, encoding="utf-8")))
    counts = {
        s: sum(g["status"] == s for g in gold)
        for s in ("accepted", "edited", "not_bio")
    }
    print(f"\n{len(gold)} reviewed: {counts}")


if __name__ == "__main__":
    main()
