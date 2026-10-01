"""The Qwen student (GGUF Q4_K_M) as an HTTP API on EC2, with llama.cpp.

The same contract as the BERT API, so the page treats both alike:
  POST any path with {"text": "..."}  ->  {"data": ..., "spans": ..., "ms": ..., "model": ...}
  GET /health                         ->  {"ok": true}

A JSON grammar built from the schema forces valid JSON, as in eval_gguf.py. The Qwen writes
the JSON itself and returns no spans, so the extracted values are looked up in the text
afterwards (bert_data.align): the page can then highlight them as it does for the BERT.

API Gateway adds a secret header (x-origin-verify); requests without it get a 403.
"""

import hmac
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from llama_cpp import Llama

from bert_data import align
from common import SCHEMA, SYSTEM_STUDENT, messages, parse_json

PORT = int(os.environ.get("PORT", "8080"))
ORIGIN_SECRET = os.environ.get("ORIGIN_SECRET", "")
MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/model/student-q4_k_m.gguf")
MODEL_NAME = os.path.basename(MODEL_PATH)
MAX_CHARS = 2000
# The grammar forces valid JSON, but checking every token against it is slow with Qwen's
# 151k-token vocabulary. The fine-tuned student writes valid JSON on its own, and the
# answer is validated against the schema either way
JSON_GRAMMAR = os.environ.get("JSON_GRAMMAR", "1") != "0"

llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=os.cpu_count(), verbose=False)
lock = threading.Lock()  # one generation at a time: a llama.cpp context is not thread-safe


def extract(text):
    t0 = time.perf_counter()
    grammar = {"response_format": {"type": "json_object", "schema": SCHEMA}} if JSON_GRAMMAR else {}
    with lock:
        out = llm.create_chat_completion(
            messages=messages(text, SYSTEM_STUDENT), temperature=0, max_tokens=300, **grammar
        )
    ms = round((time.perf_counter() - t0) * 1000)
    data = parse_json(out["choices"][0]["message"]["content"])
    spans = align(text, data)[0] if data else []
    return data, spans, ms


class Handler(BaseHTTPRequestHandler):
    def _json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ok": True, "model": MODEL_NAME})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        sent = self.headers.get("x-origin-verify", "")
        if ORIGIN_SECRET and not hmac.compare_digest(sent.encode(), ORIGIN_SECRET.encode()):
            self._json(403, {"error": "forbidden"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            text = json.loads(self.rfile.read(length).decode("utf-8"))["text"]
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError):
            self._json(400, {"error": "send a JSON body with a 'text' field"})
            return
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_CHARS:
            self._json(400, {"error": f"'text' must be between 1 and {MAX_CHARS} characters"})
            return

        data, spans, ms = extract(text)
        if data is None:
            self._json(502, {"error": "the model's JSON does not match the schema", "ms": ms})
            return
        self._json(200, {"data": data, "spans": spans, "ms": ms, "model": MODEL_NAME})

    def log_message(self, fmt, *args):  # one line per request in journalctl, without the body
        print(f"{self.address_string()} {fmt % args}", flush=True)


if __name__ == "__main__":
    print(f"listening on :{PORT} with {MODEL_NAME}, {os.cpu_count()} threads, "
          f"JSON grammar {'on' if JSON_GRAMMAR else 'off'}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
