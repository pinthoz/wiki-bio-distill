"""Labels the biographies with an open-source teacher running on the Colab GPU.

Produces the same data/labels.jsonl as label.py (Bedrock), so the rest of the notebook does not change.
Usage in Colab:
  !python label_local.py --model Qwen/Qwen2.5-7B-Instruct --limit 50   # trial run
  !python label_local.py --model Qwen/Qwen2.5-7B-Instruct              # everything (resumes where it stopped)
"""

import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import SYSTEM_TEACHER, messages, parse_json


def load_teacher(name: str, four_bit: bool):
    tokenizer = AutoTokenizer.from_pretrained(name)
    kwargs = {"device_map": "auto"}
    if four_bit:  # 7B in 4 bits takes ~5 GB: fits on a T4 with room for large batches
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@torch.no_grad()
def label_batch(model, tokenizer, rows, max_new_tokens=300):
    prompts = [
        tokenizer.apply_chat_template(
            messages(r["text"], SYSTEM_TEACHER),
            tokenize=False,
            add_generation_prompt=True,
        )
        for r in rows
    ]
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    new = out[:, enc["input_ids"].shape[1] :]
    texts = tokenizer.batch_decode(new, skip_special_tokens=True)
    results = []
    for r, raw, ids, mask in zip(rows, texts, new, enc["attention_mask"]):
        res = {
            "id": r["id"],
            "tokens_in": int(mask.sum()),
            "tokens_out": int((ids != tokenizer.pad_token_id).sum()),
        }
        label = parse_json(raw)
        if label is None:
            res.update(error="invalid JSON or does not match the schema", raw=raw[:500])
        else:
            res["label"] = label
        results.append(res)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--inp", default="data/raw.jsonl")
    ap.add_argument("--out", default="data/labels.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.inp, encoding="utf-8")]
    done = set()
    if os.path.exists(args.out):  # resume after a Colab disconnect
        done = {json.loads(line)["id"] for line in open(args.out, encoding="utf-8")}
    todo = [r for r in rows if r["id"] not in done][: args.limit]
    todo.sort(
        key=lambda r: len(r["text"])
    )  # batches of similar-length texts: less padding, faster
    print(f"{len(done)} already labeled, {len(todo)} to label with {args.model}")
    if not todo:
        return

    model, tokenizer = load_teacher(args.model, four_bit=not args.no_4bit)
    t0, errors = time.time(), 0
    with open(args.out, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), args.batch_size):
            for res in label_batch(model, tokenizer, todo[i : i + args.batch_size]):
                errors += "error" in res
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()  # each batch is saved: a disconnect only loses the current batch
            n = min(i + args.batch_size, len(todo))
            rate = n / (time.time() - t0)
            print(
                f"{n}/{len(todo)}  errors={errors}  {rate:.1f} ex/s  ~{(len(todo) - n) / rate / 60:.0f} min left"
            )


if __name__ == "__main__":
    main()
