import base64
import io
import json
import os
import time

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from bert_data import LABELS, spans_from_bio, spans_to_json

MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/model")
MODEL_FILE = os.environ.get("MODEL_FILE", "model.onnx")  # set per Lambda in infra/api_bert.tf
# s3://bucket/key of the model. Reading a big file from the container image ran at ~6 MB/s on
# a cold start (435 MB: over a minute); a parallel S3 download into memory is much faster
MODEL_S3 = os.environ.get("MODEL_S3")
MAX_CHARS = 2000


def load_model():
    """The model as bytes from S3 (in the Lambda), or its path on disk (locally)."""
    if not MODEL_S3:
        return f"{MODEL_DIR}/{MODEL_FILE}"
    import boto3
    from boto3.s3.transfer import TransferConfig

    bucket, key = MODEL_S3.removeprefix("s3://").split("/", 1)
    buf = io.BytesIO()
    config = TransferConfig(multipart_chunksize=16 * 1024**2, max_concurrency=16)
    boto3.client("s3").download_fileobj(bucket, key, buf, Config=config)
    return buf.getvalue()


# --- Cold start ---
t0 = time.perf_counter()
tokenizer = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")
tokenizer.enable_truncation(max_length=384)
model = load_model()
t1 = time.perf_counter()
session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
del model
print(f"model {MODEL_FILE}: loaded in {t1 - t0:.1f} s, session in {time.perf_counter() - t1:.1f} s")
INPUTS = {i.name for i in session.get_inputs()}
with open(f"{MODEL_DIR}/lemma_map.json", encoding="utf-8") as f:
    LEMMA_MAP = json.load(f)


def response(status, payload):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload, ensure_ascii=False),
    }


def handler(event, context):
    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw).decode()
    try:
        text = json.loads(raw)["text"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return response(400, {"error": "send a JSON body with a 'text' field"})
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_CHARS:
        return response(
            400, {"error": f"'text' must be between 1 and {MAX_CHARS} characters"}
        )

    t0 = time.perf_counter()
    enc = tokenizer.encode(text)
    feeds = {
        "input_ids": np.array([enc.ids], dtype=np.int64),
        "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
        "token_type_ids": np.array([enc.type_ids], dtype=np.int64),
    }
    logits = session.run(None, {k: v for k, v in feeds.items() if k in INPUTS})[0][0]
    tags = [LABELS[i] for i in logits.argmax(-1)]
    # special tokens ([CLS], [SEP]) become (0, 0) so that spans_from_bio skips them
    offsets = [
        (0, 0) if special else off
        for off, special in zip(enc.offsets, enc.special_tokens_mask)
    ]
    spans = spans_from_bio(text, offsets, tags)
    data = spans_to_json(spans, LEMMA_MAP)
    ms = round((time.perf_counter() - t0) * 1000)
    return response(
        200, {"data": data, "spans": spans, "ms": ms, "model": MODEL_FILE}
    )
