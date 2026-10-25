"""Simple local web UI (standard library only). Run: python web_ui.py  then open http://127.0.0.1:8000

/api/chat streams newline-delimited JSON (one event per line) while the agent works, so the tool-call trace appears
LIVE in the browser even when the local model needs a minute per step (needed for the demo video).
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config
from agents.finance_agent import FinanceAgent, check_ollama
from tools import TOOL_NAMES

HOST, PORT = "127.0.0.1", 8000
AGENT = None
LOCK = threading.Lock()
_CLIENT_GONE = (ConnectionError, BrokenPipeError, OSError)     # browser tab closed / reload (WinError 10053 on Windows)


def get_agent():
    global AGENT
    if AGENT is None:
        AGENT = FinanceAgent()
    return AGENT


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, content_type="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, default=str).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except _CLIENT_GONE:
            pass            # the browser closed the connection before we answered - nothing to do

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = (config.BASE_DIR / "web" / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
        elif self.path == "/api/info":
            self._send(200, {"model": config.MODEL, "tools": TOOL_NAMES, "max_steps": config.MAX_STEPS})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"error": "invalid JSON"})
        if self.path == "/api/chat":
            message = payload.get("message", "") if isinstance(payload, dict) else ""
            self._stream_chat(message)
        elif self.path == "/api/reset":
            with LOCK:
                get_agent().reset()
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "not found"})

    def _stream_chat(self, message):
        alive = True

        def push(obj):
            nonlocal alive
            if not alive:
                return
            try:
                self.wfile.write((json.dumps(obj, default=str, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()
            except _CLIENT_GONE:
                alive = False       # keep running: the agent may already have saved data, it must finish cleanly

        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
        except _CLIENT_GONE:
            alive = False
        self.close_connection = True
        with LOCK:
            result = get_agent().chat(message, on_event=push)
        push({"type": "done", "answer": result.answer, "steps": result.steps,
              "hit_step_limit": result.hit_step_limit, "error": result.error, "tool_calls": len(result.events)})

    def log_message(self, fmt, *args):      # keep the console quiet
        pass


def main():
    ok, message = check_ollama()
    print(message)
    if not ok:
        raise SystemExit(1)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Web UI running at http://{HOST}:{PORT}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
