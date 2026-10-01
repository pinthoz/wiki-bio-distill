"""Predictions of the quantized Qwen student (GGUF) on the test set, on CPU with llama.cpp.

The same as the last cell of notebook/train.ipynb, for running on a laptop. The JSON grammar
from the schema forces valid JSON, as in production.

Usage:
  pip install llama-cpp-python
  python eval_gguf.py --model student-q4_k_m.gguf
  python evaluate.py preds/student.jsonl preds/student-q4.jsonl
"""

import argparse
import json
import time

from llama_cpp import Llama

from common import SCHEMA, SYSTEM_STUDENT, messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="student-q4_k_m.gguf")
    ap.add_argument("--out", default="preds/student-q4.jsonl")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (default: all)")
    args = ap.parse_args()

    llm = Llama(model_path=args.model, n_ctx=2048, n_threads=args.threads, verbose=False)
    test = [json.loads(l) for l in open("data/test.jsonl", encoding="utf-8")]
    with open(args.out, "w", encoding="utf-8") as f:
        for i, t in enumerate(test, 1):
            t0 = time.perf_counter()
            out = llm.create_chat_completion(
                messages=messages(t["text"], SYSTEM_STUDENT),
                temperature=0,
                max_tokens=300,
                response_format={"type": "json_object", "schema": SCHEMA},
            )
            ms = round((time.perf_counter() - t0) * 1000)
            raw = out["choices"][0]["message"]["content"]
            f.write(json.dumps({"id": t["id"], "raw": raw, "ms": ms}, ensure_ascii=False) + "\n")
            print(f"{i}/{len(test)}  {ms} ms", flush=True)


if __name__ == "__main__":
    main()
