"""Labels the biographies with a teacher on Bedrock (Converse API + tool use).

The alternative to label_local.py (an open-source teacher on Colab): it writes the same
data/labels.jsonl, so the rest of the pipeline does not change.

Usage:
  export MODEL_ID="<model or inference profile ID, copied from the Bedrock console>"
  export PRICE_IN=<USD per 1M input tokens>  PRICE_OUT=<USD per 1M output tokens>
  python label.py --inp data/raw.jsonl --out data/labels.jsonl --limit 50   # cost trial
  python label.py --inp data/raw.jsonl --out data/labels.jsonl              # everything (resumes where it stopped)
"""

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.config import Config
from pydantic import ValidationError

from common import SCHEMA, SYSTEM_TEACHER, Biography

MODEL_ID = os.environ["MODEL_ID"]
PRICE_IN = float(os.environ.get("PRICE_IN", "0"))
PRICE_OUT = float(os.environ.get("PRICE_OUT", "0"))

bedrock = boto3.client(
    "bedrock-runtime",
    config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),  # backs off on throttling
)

# Forcing the model to "call a tool" is the most reliable way to get JSON that fits the schema
TOOL = {
    "toolSpec": {
        "name": "save_biography",
        "description": "Saves the data extracted from the biography.",
        "inputSchema": {"json": SCHEMA},
    }
}


def label_one(row: dict) -> dict:
    resp = bedrock.converse(
        modelId=MODEL_ID,
        system=[{"text": SYSTEM_TEACHER}],
        messages=[{"role": "user", "content": [{"text": row["text"]}]}],
        toolConfig={"tools": [TOOL], "toolChoice": {"tool": {"name": "save_biography"}}},
        inferenceConfig={"temperature": 0, "maxTokens": 600},
    )
    usage = resp["usage"]
    out = {
        "id": row["id"],
        "tokens_in": usage["inputTokens"],
        "tokens_out": usage["outputTokens"],
    }
    blocks = resp["output"]["message"]["content"]
    tool_input = next((b["toolUse"]["input"] for b in blocks if "toolUse" in b), None)
    try:
        out["label"] = Biography.model_validate(tool_input).model_dump()
    except ValidationError as e:
        out["error"] = str(e)[:300]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/raw.jsonl")
    ap.add_argument("--out", default="data/labels.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.inp, encoding="utf-8")]
    done = set()
    if os.path.exists(args.out):  # resume: never pay twice for the same example
        done = {json.loads(line)["id"] for line in open(args.out, encoding="utf-8")}
    todo = [r for r in rows if r["id"] not in done][: args.limit]
    print(f"{len(done)} already labeled, {len(todo)} to label with {MODEL_ID}")

    lock, tin, tout, errors = threading.Lock(), 0, 0, 0
    with open(args.out, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(label_one, r) for r in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            with lock:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                f.flush()
                tin, tout = tin + res["tokens_in"], tout + res["tokens_out"]
                errors += "error" in res
            if i % 50 == 0 or i == len(todo):
                cost = tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
                print(f"{i}/{len(todo)}  errors={errors}  tokens={tin}+{tout}  cost≈${cost:.3f}")

    if todo:
        per = (tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT) / len(todo)
        print(
            f"average cost per example ≈ ${per:.5f}  →  projection for {len(rows)}: "
            f"≈ ${per * len(rows):.2f}"
        )


if __name__ == "__main__":
    main()
