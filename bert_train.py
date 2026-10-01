"""Second student: BERT for token classification, trained locally (notebook 03 without Colab).

Needs data/raw, train, val and test.jsonl. Trains, predicts the gold test set into
preds/bert.jsonl, prints the BERT errors and exports to ONNX (validated against PyTorch).
Usage: python bert_train.py
       python bert_train.py --model PORTULAN/albertina-100m-portuguese-ptpt-encoder --out run-albertina
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

import bert_data
from bert_data import LABEL2ID, LABELS, spans_from_bio, spans_to_json
from common import FIELDS, as_set, parse_json

MAX_LEN = 384


def encode(ex, tokenizer):
    enc = tokenizer(
        ex["text"], truncation=True, max_length=MAX_LEN, return_offsets_mapping=True
    )
    labels = []
    for s, e in enc["offset_mapping"]:
        if s == e:  # [CLS], [SEP]: ignored in the loss
            labels.append(-100)
            continue
        tag = "O"
        for sp in ex["spans"]:
            if s >= sp["start"] and e <= sp["end"]:  # token inside the span
                tag = ("B-" if s == sp["start"] else "I-") + sp["label"]
                break
        labels.append(LABEL2ID[tag])
    enc["labels"] = labels
    enc.pop("offset_mapping")
    return enc


def train(args, tokenizer):
    data = load_dataset(
        "json",
        data_files={
            "train": "data/bert_train.jsonl",
            "validation": "data/bert_val.jsonl",
        },
    )
    tok_data = data.map(
        lambda ex: encode(ex, tokenizer), remove_columns=data["train"].column_names
    )

    # check on one example that the tags land on the right words
    ex = tok_data["train"][0]
    for t, l in zip(
        tokenizer.convert_ids_to_tokens(ex["input_ids"])[:40], ex["labels"][:40]
    ):
        print(f"{t:>15}  {LABELS[l] if l >= 0 else '-'}")

    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=LABEL2ID,
    )
    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=torch.cuda.is_available(),
        report_to="none",
        logging_steps=20,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=tok_data["train"],
        eval_dataset=tok_data["validation"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )
    checkpoints = glob.glob(f"{args.out}/checkpoint-*")
    if checkpoints and args.fresh:
        for c in checkpoints:
            shutil.rmtree(c)
        checkpoints = []
    if checkpoints:  # resume an interrupted run; a finished one is not trained again
        print(f"resuming from {max(checkpoints, key=os.path.getmtime)}: "
              "a finished run is only re-evaluated (use --fresh to train from scratch)")
    trainer.train(resume_from_checkpoint=bool(checkpoints))
    trainer.save_model(f"{args.out}/best")
    tokenizer.save_pretrained(f"{args.out}/best")
    return model


@torch.no_grad()
def predict(model, tokenizer, test):
    lemma_map = json.load(open("data/lemma_map.json", encoding="utf-8"))
    model.eval()
    Path("preds").mkdir(exist_ok=True)
    with open("preds/bert.jsonl", "w", encoding="utf-8") as f:
        for t in test:
            t0 = time.perf_counter()
            enc = tokenizer(
                t["text"],
                truncation=True,
                max_length=MAX_LEN,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = enc.pop("offset_mapping")[0].tolist()
            logits = model(**{k: v.to(model.device) for k, v in enc.items()}).logits[0]
            tags = [LABELS[i] for i in logits.argmax(-1).tolist()]
            pred = spans_to_json(spans_from_bio(t["text"], offsets, tags), lemma_map)
            ms = round((time.perf_counter() - t0) * 1000)
            row = {"id": t["id"], "raw": json.dumps(pred, ensure_ascii=False), "ms": ms}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def show_errors(test, n=10):
    """Model errors (wrong span) vs. post-processing errors (bad date or lemma)."""
    preds = {
        p["id"]: parse_json(p["raw"])
        for p in map(json.loads, open("preds/bert.jsonl", encoding="utf-8"))
    }
    shown = 0
    for t in test:
        p = preds[t["id"]] or {}
        wrong = [f for f in FIELDS if as_set(p.get(f)) != as_set(t["gold"].get(f))]
        if wrong and shown < n:
            shown += 1
            print(t["text"][:200], "…")
            for f in wrong:
                print(f"  {f}: gold={t['gold'].get(f)!r}  bert={p.get(f)!r}")
            print()


def export_onnx(args, tokenizer):
    out = Path(args.out) / "onnx"
    out.mkdir(parents=True, exist_ok=True)
    # eager attention: the legacy exporter does not translate SDPA attention correctly
    cpu_model = AutoModelForTokenClassification.from_pretrained(
        f"{args.out}/best", attn_implementation="eager"
    ).eval()
    # Two texts of different lengths (padded): batch and length stay dynamic in the export
    sample = tokenizer(
        ["Uma frase de exemplo.", "Outra frase de exemplo, um pouco mais comprida."],
        padding=True,
        return_tensors="pt",
    )
    names = list(sample.keys())
    batch, seq = torch.export.Dim("batch"), torch.export.Dim("seq", max=MAX_LEN)
    # torch.export-based exporter: the legacy one bakes the attention mask of the sample
    # into the graph, and then only that exact input length gives the right result
    torch.onnx.export(
        cpu_model,
        args=(),
        kwargs=dict(sample),
        f=str(out / "model.onnx"),
        input_names=names,
        output_names=["logits"],
        dynamic_shapes={n: {0: batch, 1: seq} for n in names},
        opset_version=18,
        dynamo=True,
        external_data=False,  # a single model.onnx file, as the Lambda expects
    )
    tokenizer.save_pretrained(out)
    (out / "lemma_map.json").write_bytes(Path("data/lemma_map.json").read_bytes())

    # validation: PyTorch and ONNX must give the same logits, also on real texts of other lengths
    test = [json.loads(l) for l in open("data/test.jsonl", encoding="utf-8")]
    texts = ["Uma frase de exemplo."] + [t["text"] for t in test[:50]]
    encs = [
        tokenizer(t, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        for t in texts
    ]
    with torch.no_grad():
        refs = [cpu_model(**enc).logits.numpy() for enc in encs]

    def check(path):
        """Logit difference, and the share of tokens / whole texts with the same tag."""
        sess = ort.InferenceSession(str(path))
        diff, same_texts, same_tokens, n_tokens = 0.0, 0, 0, 0
        for enc, ref in zip(encs, refs):
            got = sess.run(None, {k: v.numpy() for k, v in enc.items()})[0]
            diff = max(diff, float(abs(ref - got).max()))
            same = ref.argmax(-1) == got.argmax(-1)
            same_texts += bool(same.all())
            same_tokens, n_tokens = same_tokens + int(same.sum()), n_tokens + same.size
        size = path.stat().st_size / 1e6
        print(
            f"{path.name} ({size:.0f} MB): max difference vs PyTorch {diff:.2e}, "
            f"same tags on {same_tokens / n_tokens:.2%} of tokens "
            f"and {same_texts}/{len(texts)} whole texts"
        )
        return same_tokens / n_tokens

    check(out / "model.onnx")

    # A smaller model: the fp32 one (435 MB) is too slow to load within Lambda's 10 s init.
    # Try a few quantizations and keep the one whose tags agree best with PyTorch
    def dynamic_int8(src, dst):
        # int8 weights AND int8 activations (one scale per tensor, computed per request)
        quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)

    def weight_only(bits):
        # int4/int8 weights in blocks of 32, each with its own scale; activations stay
        # in fp32, so BERT's outlier activations are not squashed
        def run(src, dst):
            import inspect

            from onnxruntime.quantization import matmul_nbits_quantizer as mnq

            # The number of bits moved between onnxruntime versions: an argument of the
            # quantizer, of its config object, or not configurable (4 bits only)
            model = onnx.load(str(src))
            if "bits" in inspect.signature(mnq.MatMulNBitsQuantizer).parameters:
                q = mnq.MatMulNBitsQuantizer(model, bits=bits, block_size=32, is_symmetric=True)
            elif "bits" in inspect.signature(mnq.DefaultWeightOnlyQuantConfig).parameters:
                cfg = mnq.DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, bits=bits)
                q = mnq.MatMulNBitsQuantizer(model, algo_config=cfg)
            elif bits == 4:
                q = mnq.MatMulNBitsQuantizer(model, block_size=32, is_symmetric=True)
            else:
                raise RuntimeError(f"this onnxruntime only quantizes weights to 4 bits, not {bits}")
            q.process()
            q.model.save_model_to_file(str(dst), use_external_data_format=False)

        return run

    candidates = {
        "dynamic-int8": dynamic_int8,
        "weights-int8": weight_only(8),
        "weights-int4": weight_only(4),
    }
    best = None
    for name, quantize in candidates.items():
        dst = out / f"model.quant.{name}.onnx"
        print(f"\n[{name}]")
        try:
            quantize(out / "model.onnx", dst)
            score = check(dst)
        except Exception as e:
            print(f"failed: {str(e)[:150]}")
            continue
        if best is None or score > best[0]:
            best = (score, name, dst)
    if best:
        best[2].replace(out / "model.quant.onnx")
        print(f"\nkept '{best[1]}' as model.quant.onnx")
    for f in out.glob("model.quant.*.onnx"):
        f.unlink()
    (out / "model.int8.onnx").unlink(missing_ok=True)  # left over from earlier runs

    # What matters is the extraction: F1 on the gold test set, PyTorch vs ONNX fp32 vs quantized
    predict_onnx(out / "model.onnx", tokenizer, test, "preds/bert-onnx.jsonl")
    predict_onnx(out / "model.quant.onnx", tokenizer, test, "preds/bert-quant.jsonl")
    preds = ["preds/bert.jsonl", "preds/bert-onnx.jsonl", "preds/bert-quant.jsonl"]
    subprocess.run([sys.executable, "evaluate.py", *preds])
    print(f"ONNX models in {out}/")


def predict_onnx(path, tokenizer, test, out_path):
    """Same as predict(), but with an ONNX Runtime session on the CPU, as in the Lambda."""
    lemma_map = json.load(open("data/lemma_map.json", encoding="utf-8"))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}
    with open(out_path, "w", encoding="utf-8") as f:
        for t in test:
            t0 = time.perf_counter()
            enc = tokenizer(
                t["text"],
                truncation=True,
                max_length=MAX_LEN,
                return_offsets_mapping=True,
                return_tensors="np",
            )
            offsets = enc.pop("offset_mapping")[0].tolist()
            logits = sess.run(None, {k: v for k, v in enc.items() if k in names})[0][0]
            tags = [LABELS[i] for i in logits.argmax(-1).tolist()]
            pred = spans_to_json(spans_from_bio(t["text"], offsets, tags), lemma_map)
            ms = round((time.perf_counter() - t0) * 1000)
            row = {"id": t["id"], "raw": json.dumps(pred, ensure_ascii=False), "ms": ms}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="neuralmind/bert-base-portuguese-cased")
    ap.add_argument("--out", default="run-bert")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--export-only", action="store_true", help="skip training; export <out>/best"
    )
    ap.add_argument(
        "--fresh", action="store_true", help="delete old checkpoints and train from scratch"
    )
    args = ap.parse_args()
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

    if args.export_only:
        export_onnx(args, AutoTokenizer.from_pretrained(f"{args.out}/best"))
        return

    bert_data.main()  # JSON -> spans; prints the share of fully aligned examples
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = train(args, tokenizer)

    test = [json.loads(l) for l in open("data/test.jsonl", encoding="utf-8")]
    predict(model, tokenizer, test)
    others = [
        p
        for p in ("preds/base.jsonl", "preds/student.jsonl", "preds/student-q4.jsonl")
        if Path(p).exists()
    ]
    subprocess.run([sys.executable, "evaluate.py", *others, "preds/bert.jsonl"])
    show_errors(test)
    export_onnx(args, tokenizer)


if __name__ == "__main__":
    main()
