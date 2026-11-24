# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local dashboard server.

A small standard-library HTTP server that serves the static dashboard and a
JSON snapshot of the running pipeline. It binds to 127.0.0.1 by default, serves
only a fixed list of files, and has no third-party or CDN dependencies.
"""

from __future__ import annotations

import json
import math
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}
MAX_BODY = 1024
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'; base-uri 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


def _finite(obj):
    """JSON has no NaN or infinity; send them as null."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def make_handler(pipeline, refresh_ms: int, host: str = "127.0.0.1"):
    allowed_hosts = {"127.0.0.1", "localhost", "::1", host.lower()}

    class Handler(BaseHTTPRequestHandler):
        server_version = "ScreeningTools"
        sys_version = ""

        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in {**SECURITY_HEADERS, **(extra or {})}.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = 200, extra: dict | None = None) -> None:
            self._send(status, json.dumps(_finite(obj), default=str, allow_nan=False).encode("utf-8"), "application/json", extra)

        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
            if host in allowed_hosts:
                return True
            self._json({"error": "forbidden host"}, HTTPStatus.FORBIDDEN)
            return False

        def do_GET(self):  # noqa: N802
            if not self._host_ok():
                return
            path = self.path.split("?", 1)[0]
            if path in STATIC_FILES:
                name, ctype = STATIC_FILES[path]
                self._send(200, (STATIC_DIR / name).read_bytes(), ctype)
            elif path == "/api/snapshot":
                snap = pipeline.snapshot()
                snap["refresh_ms"] = refresh_ms
                snap["server_time"] = time.time()
                self._json(snap)
            elif path == "/api/summary":
                summary = pipeline.out_dir / "summary.json"
                if summary.is_file():
                    self._send(
                        200,
                        summary.read_bytes(),
                        "application/json",
                        {"Content-Disposition": f'attachment; filename="{pipeline.cfg["run"]["run_id"]}-summary.json"'},
                    )
                else:
                    self._json({"error": "the summary is written when the run finishes"}, HTTPStatus.NOT_FOUND)
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self):  # noqa: N802
            if not self._host_ok():
                return
            path = self.path.split("?", 1)[0]
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY:
                self._json({"error": "bad request"}, HTTPStatus.BAD_REQUEST)
                return
            raw = self.rfile.read(length) if length else b"{}"
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                self._json({"error": "expected application/json"}, HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
                return
            try:
                body = json.loads(raw)
            except ValueError:
                self._json({"error": "bad json"}, HTTPStatus.BAD_REQUEST)
                return
            if path == "/api/ui-lag":
                lag = body.get("lag_ms") if isinstance(body, dict) else None
                if isinstance(lag, (int, float)) and 0 <= lag < 600_000:
                    pipeline.telemetry.record_ui_lag(lag / 1000.0)
                    self._json({"ok": True})
                else:
                    self._json({"error": "lag_ms must be a number of milliseconds"}, HTTPStatus.BAD_REQUEST)
            elif path == "/api/stop":
                pipeline.request_stop()
                self._json({"ok": True})
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    return Handler


class DashboardServer:
    def __init__(self, pipeline, host: str, port: int, refresh_ms: int):
        self.httpd = ThreadingHTTPServer((host, port), make_handler(pipeline, refresh_ms, host))
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="dashboard", daemon=True)

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}/"

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
