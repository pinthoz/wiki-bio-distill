"""Collects biography introductions from the Portuguese Wikipedia.

Usage: python collect.py --n 3000 --out data/raw.jsonl
"""

import argparse
import json
import random
import re
from pathlib import Path

# "X (Lisboa, 3 de maio de 1920 — Porto, 1 de abril de 1990) foi um escritor..."
# Inside the parentheses we require "Place, date": that is what separates people from events
# such as "A Batalha de Aljubarrota (14 de agosto de 1385) foi uma batalha...".
# An appositive is allowed between the parentheses and the verb: "(...), mais conhecida como X, foi uma..."
BIO_RE = re.compile(
    r"^[^()\n]{3,80}"
    r"\((?=[^)]*[A-ZÀ-Ú][^,()]{1,40}, (?:\d{1,2}º? de )?[a-zç]+ de \d{3,4})[^)]{4,200}\)"
    r"[^.]{0,120}?\s(foi|é|era)\s(um|uma|o|a)\b"
)


def first_paragraph(text: str) -> str:
    for para in text.split("\n"):
        para = para.strip()
        if len(para) > 80:
            return para
    return ""


def is_biography(para: str) -> bool:
    return 150 <= len(para) <= 1200 and bool(BIO_RE.match(para))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument(
        "--scan", type=int, default=400_000, help="maximum number of articles to scan"
    )
    ap.add_argument("--out", default="data/raw.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import (
        load_dataset,
    )  # imported here only: tests of the functions above do not need it

    stream = load_dataset(
        "wikimedia/wikipedia", "20231101.pt", split="train", streaming=True
    )
    stream = stream.shuffle(seed=args.seed, buffer_size=10_000)

    random.seed(args.seed)
    out, seen = [], set()
    for i, art in enumerate(stream):
        if i >= args.scan or len(out) >= args.n:
            break
        para = first_paragraph(art["text"])
        if is_biography(para) and art["title"] not in seen:
            seen.add(art["title"])
            out.append(
                {
                    "id": art["id"],
                    "title": art["title"],
                    "url": art["url"],
                    "text": para,
                }
            )
        if i % 20_000 == 0:
            print(f"{i} articles read, {len(out)} biographies")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"saved {len(out)} biographies to {args.out}")


if __name__ == "__main__":
    main()
