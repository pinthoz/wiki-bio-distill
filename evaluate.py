"""Compare predictions with the gold test set.

Usage:
  python evaluate.py                                  # just the teacher (comes in test.jsonl)
  python evaluate.py preds/base.jsonl preds/student.jsonl
Each prediction file has one line per example: {"id": ..., "raw": "<generated text>", "ms": 812}
"""

import json
import sys

from common import FIELDS, evaluate, parse_json


def load_preds(path):
    return {p["id"]: p for p in map(json.loads, open(path, encoding="utf-8"))}


def main():
    test = [json.loads(l) for l in open("data/test.jsonl", encoding="utf-8")]
    golds = [t["gold"] for t in test]
    results = {"professor": evaluate([t["teacher"] for t in test], golds)}
    latency = {}
    for path in sys.argv[1:]:
        name = path.split("/")[-1].removesuffix(".jsonl")
        preds = load_preds(path)
        results[name] = evaluate(
            [
                parse_json(preds[t["id"]]["raw"]) if t["id"] in preds else None
                for t in test
            ],
            golds,
        )
        ms = sorted(p["ms"] for p in preds.values() if "ms" in p)
        if ms:
            latency[name] = (ms[len(ms) // 2], ms[int(len(ms) * 0.95) - 1])

    names = list(results)
    print(f"{'métrica':<22}" + "".join(f"{n:>14}" for n in names))
    rows = [
        ("JSON válido", lambda r: r["json_valid"]),
        ("F1 global", lambda r: r["overall"]["f1"]),
        ("precisão global", lambda r: r["overall"]["precision"]),
        ("recall global", lambda r: r["overall"]["recall"]),
    ]
    rows += [(f"F1 {f}", lambda r, f=f: r["fields"][f]["f1"]) for f in FIELDS]
    for label, get in rows:
        print(f"{label:<22}" + "".join(f"{get(results[n]):>14.3f}" for n in names))
    for n, (p50, p95) in latency.items():
        print(f"latência {n}: p50={p50} ms  p95={p95} ms")
    json.dump(results, open("results.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
