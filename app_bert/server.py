"""The BERT API as a plain HTTP server, for running it on EC2 next to the Lambda.

It wraps the same handler as the Lambda (app.py), so both answer exactly the same JSON.
The model loads once, when the server starts, and stays in memory: no cold starts.

  POST any path with {"text": "..."}  ->  {"data": ..., "spans": ..., "ms": ..., "model": ...}
  GET /health                         ->  {"ok": true}

API Gateway adds a secret header to every request it forwards (x-origin-verify). The server
refuses requests without it, so nobody can skip the API key by calling the instance directly.
"""

import hmac
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import app  # loads the tokenizer and the ONNX model, like a Lambda cold start

PORT = int(os.environ.get("PORT", "8080"))
ORIGIN_SECRET = os.environ.get("ORIGIN_SECRET", "")


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, headers=None):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(status)
        for k, v in (headers or {"Content-Type": "application/json"}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, json.dumps({"ok": True, "model": app.MODEL_FILE}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        sent = self.headers.get("x-origin-verify", "")
        if ORIGIN_SECRET and not hmac.compare_digest(sent.encode(), ORIGIN_SECRET.encode()):
            self._send(403, json.dumps({"error": "forbidden"}))
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode("utf-8", errors="replace")
        res = app.handler({"body": body}, None)
        self._send(res["statusCode"], res["body"], res["headers"])

    def log_message(self, fmt, *args):  # one line per request in journalctl, without the body
        print(f"{self.address_string()} {fmt % args}", flush=True)


if __name__ == "__main__":
    print(f"listening on :{PORT} with {app.MODEL_FILE}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
