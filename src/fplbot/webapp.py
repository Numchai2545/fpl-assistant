"""Serve the local dashboard with one browser-based refresh button.

The server binds to localhost only. It gives a non-technical user a normal web
entry point while keeping every FPL decision manual and every credential out of
the project.
"""
from __future__ import annotations

import html
import json
import os
import subprocess
import sys
import threading
import urllib.request
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def build_environment(project_dir: str | Path) -> dict[str, str]:
    """Return the environment required by the src-layout package subprocess."""
    env = os.environ.copy()
    source_dir = str(Path(project_dir).resolve() / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = source_dir + (os.pathsep + existing if existing else "")
    env["PYTHONIOENCODING"] = "utf-8"
    return env


class DashboardHandler(SimpleHTTPRequestHandler):
    server_version = "FPLAssistant/1.0"
    build_lock = threading.Lock()

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            payload = b"fpl-assistant"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/":
            self.send_response(302)
            self.send_header("Location", "/index.html")
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/refresh":
            self.send_error(404)
            return

        wants_json = "application/json" in self.headers.get("Accept", "")
        if not self.build_lock.acquire(blocking=False):
            if wants_json:
                self._send_json(409, {"ok": False, "error": "กำลังอัปเดตข้อมูลอยู่แล้ว"})
            else:
                self.send_error(409, "Build already running")
            return

        command = [sys.executable, "-m", "fplbot"]
        config_path = getattr(self.server, "config_path", None)
        if config_path:
            command.extend(["--config", config_path])
        command.append("build")
        project_dir = getattr(self.server, "project_dir")
        try:
            result = subprocess.run(
                command, cwd=project_dir, env=build_environment(project_dir),
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=300,
            )
        except subprocess.TimeoutExpired:
            result = subprocess.CompletedProcess(command, 124, "", "การอัปเดตใช้เวลาเกิน 5 นาที")
        finally:
            self.build_lock.release()
        if result.returncode == 0:
            if wants_json:
                self._send_json(200, {"ok": True})
            else:
                self.send_response(303)
                self.send_header("Location", "/index.html?updated=1")
                self.end_headers()
            return

        detail = html.escape((result.stderr or result.stdout or "Unknown error")[-4000:])
        if wants_json:
            self._send_json(500, {"ok": False, "error": html.unescape(detail)})
            return
        body = f"""<!doctype html><html lang='th'><meta charset='utf-8'>
        <meta name='viewport' content='width=device-width,initial-scale=1'>
        <title>อัปเดตไม่สำเร็จ</title><style>
        body{{font-family:system-ui;margin:40px auto;max-width:760px;padding:0 18px}}
        pre{{white-space:pre-wrap;background:#f4eef7;padding:16px;border-radius:8px}}
        a{{color:#6d28a8}}</style><h1>อัปเดตข้อมูลไม่สำเร็จ</h1>
        <p>รายงานเดิมยังเปิดดูได้ ลองตรวจอินเทอร์เน็ตแล้วกดกลับไปอัปเดตใหม่</p>
        <pre>{detail}</pre><p><a href='/index.html'>กลับไปหน้ารายงาน</a></p></html>"""
        payload = body.encode("utf-8")
        self.send_response(500)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:
        # Keep the minimized launcher quiet; build failures are shown in-browser.
        return


def _is_our_server(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=1) as response:
            return response.read() == b"fpl-assistant"
    except OSError:
        return False


def serve(site_dir: Path, project_dir: Path, *, port: int = 8765,
          config_path: str | None = None, open_browser: bool = True) -> int:
    """Start the local dashboard, or reuse an instance already running."""
    url = f"http://127.0.0.1:{port}"
    if _is_our_server(url):
        if open_browser:
            webbrowser.open(url)
        return 0

    handler = partial(DashboardHandler, directory=str(site_dir.resolve()))
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError as exc:
        raise RuntimeError(f"local web port {port} is already in use") from exc
    server.project_dir = str(project_dir.resolve())
    server.config_path = config_path
    if open_browser:
        threading.Timer(0.35, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
