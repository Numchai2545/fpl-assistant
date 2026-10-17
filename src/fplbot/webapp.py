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
import time
import signal
import urllib.request
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from .config import load_config
from .copy_th import text as copy_text


def run_build(command: list[str], project_dir: str, timeout: float,
              on_line: Callable[[str], None] | None = None) -> subprocess.CompletedProcess:
    """Bound the entire build, including CBC children and network retries."""
    options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
               if os.name == "nt" else {"start_new_session": True})
    process = subprocess.Popen(command, cwd=project_dir, env=build_environment(project_dir),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", **options)
    output: list[str] = []

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output.append(line)
            if on_line:
                on_line(line)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    try:
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NO_WINDOW, timeout=10)
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=10)
        code = 124
    finally:
        reader.join(timeout=2)
        if process.stdout is not None:
            process.stdout.close()
    return subprocess.CompletedProcess(command, code, "".join(output)[-8000:],
                                        copy_text("build_timeout") if code == 124 else "")


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
        if self.path == "/refresh-status":
            started = getattr(self.server, "build_started", None)
            self._send_json(200, {"running": self.build_lock.locked(),
                                 "elapsed_seconds": int(time.monotonic() - started) if started else 0,
                                 "stage": getattr(self.server, "build_stage", copy_text("stage_fetch"))})
            return
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
        command.extend(["build", "--fresh"])
        project_dir = getattr(self.server, "project_dir")
        self.server.build_started = time.monotonic()
        self.server.build_stage = copy_text("stage_fetch")

        def progress(line: str) -> None:
            for marker, key in (("building features", "stage_features"),
                                ("solving hold", "stage_solve"),
                                ("dashboard written", "stage_output")):
                if marker in line:
                    self.server.build_stage = copy_text(key)
        try:
            result = run_build(command, project_dir,
                               getattr(self.server, "build_timeout", 300), progress)
        except OSError:
            result = subprocess.CompletedProcess(command, 1, "", "ไม่สามารถเริ่มการอัปเดตได้")
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
    server.build_timeout = float(load_config(config_path).get("planning", "build_time_budget", default=300))
    if open_browser:
        threading.Timer(0.35, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
