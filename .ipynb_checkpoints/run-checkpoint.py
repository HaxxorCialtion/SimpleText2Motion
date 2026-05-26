#!/usr/bin/env python3
"""
run.py — SimpleLove streaming demo server (cross-platform).

GET  /              -> serves ./index.html (sibling of this script)
GET  /assets/<f>    -> static .vrm / .vrma from ./assets/
POST /t2m           -> { prompt, duration?, seed? } returns binary .vrma

Layout expected next to this script:
  ./index.html
  ./assets/example.vrm
  ./assets/idle.vrma
  ./assets/0.vrma, 1.vrma   (optional)
  ./build/motion_to_vrma          (Linux/macOS)
  ./build/Release/motion_to_vrma.exe   (Windows)
  ./client/t2m_client.py
  ./scripts/start_servers.sh     (or .bat) launched separately

Prereqs (started separately):
  fused server + t2m_infer listening on 127.0.0.1:8423
"""

from __future__ import annotations

import http.server
import json
import mimetypes
import os
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path


# ===========================================================================
# Paths (cross-platform)
# ===========================================================================
ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "assets"
CLIENT_DIR = ROOT / "client"
INDEX_HTML = ROOT / "index.html"


def _bin(name: str) -> Path:
    """Resolve a native binary path per platform.

    Windows: build\\Release\\<name>.exe
    Linux/macOS: build/<name>
    """
    if os.name == "nt":
        return ROOT / "build" / "Release" / f"{name}.exe"
    return ROOT / "build" / name


MOTION_TO_VRMA_BIN = _bin("motion_to_vrma")


# ===========================================================================
# Import t2m client
# ===========================================================================
sys.path.insert(0, str(CLIENT_DIR))
try:
    from t2m_client import call_t2m  # type: ignore
except ImportError as e:
    print(f"FATAL: could not import t2m_client from {CLIENT_DIR}: {e}", file=sys.stderr)
    sys.exit(1)


# ===========================================================================
# Config
# ===========================================================================
PORT = 8765
T2M_HOST = "127.0.0.1"
T2M_PORT = 8423
DEFAULT_DURATION = 3.0
DEFAULT_SEED = 42


# ===========================================================================
# t2m pipeline
# ===========================================================================
def generate_vrma(prompt: str, duration: float, seed: int) -> bytes:
    if not MOTION_TO_VRMA_BIN.exists():
        raise FileNotFoundError(f"motion_to_vrma not found at {MOTION_TO_VRMA_BIN}")

    motion, _ = call_t2m(T2M_HOST, T2M_PORT, prompt, duration, seed)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        bin_path = td_path / "motion.bin"
        vrma_path = td_path / "motion.vrma"
        motion.astype("<f4").tofile(bin_path)
        result = subprocess.run(
            [
                str(MOTION_TO_VRMA_BIN),
                "--in", str(bin_path),
                "--out", str(vrma_path),
                "--title", prompt[:60],
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"motion_to_vrma rc={result.returncode}: {result.stderr.strip()}"
            )
        return vrma_path.read_bytes()


# ===========================================================================
# HTTP handler
# ===========================================================================
class Handler(http.server.BaseHTTPRequestHandler):
    # ---- low-level send helpers ----
    def _send_bytes(self, status: int, body: bytes, content_type: str = "application/octet-stream") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status: int, msg: str) -> None:
        self._send_bytes(status, msg.encode("utf-8"), "text/plain; charset=utf-8")

    # ---- GET ----
    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]

        if path in ("/", "/index.html"):
            if not INDEX_HTML.exists():
                self._send_text(500, f"index.html missing at {INDEX_HTML}")
                return
            self._send_bytes(
                200,
                INDEX_HTML.read_bytes(),
                "text/html; charset=utf-8",
            )
            return

        if path.startswith("/assets/"):
            rel = path[len("/assets/"):]
            # Reject traversal attempts
            if ".." in rel.split("/") or rel.startswith("/"):
                self._send_text(403, "forbidden")
                return
            asset_path = (ASSETS_DIR / rel).resolve()
            try:
                asset_path.relative_to(ASSETS_DIR.resolve())
            except ValueError:
                self._send_text(403, "forbidden")
                return
            if not asset_path.exists() or not asset_path.is_file():
                self._send_text(404, f"not found: {rel}")
                return
            suffix = asset_path.suffix.lower()
            if suffix in (".vrm", ".vrma"):
                mime = "model/gltf-binary"
            else:
                mime = mimetypes.guess_type(asset_path.name)[0] or "application/octet-stream"
            self._send_bytes(200, asset_path.read_bytes(), mime)
            return

        self._send_text(404, "not found")

    # ---- POST ----
    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/t2m":
            self._send_text(404, "not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b""
            payload = json.loads(raw.decode("utf-8")) if raw else {}
            prompt = (payload.get("prompt") or "").strip()
            duration = float(payload.get("duration") or DEFAULT_DURATION)
            seed = int(payload.get("seed") or DEFAULT_SEED)
            if not prompt:
                self._send_text(400, "empty prompt")
                return
        except (ValueError, json.JSONDecodeError) as e:
            self._send_text(400, f"bad request: {e}")
            return

        sys.stderr.write(f"[t2m] {prompt!r} dur={duration} seed={seed}\n")
        t0 = time.perf_counter()
        try:
            vrma_bytes = generate_vrma(prompt, duration, seed)
        except Exception as e:
            sys.stderr.write(f"[t2m] FAIL: {type(e).__name__}: {e}\n")
            self._send_text(500, f"{type(e).__name__}: {e}")
            return
        sys.stderr.write(
            f"[t2m] OK {len(vrma_bytes)}B in {(time.perf_counter() - t0) * 1000:.0f}ms\n"
        )
        self._send_bytes(200, vrma_bytes, "model/gltf-binary")

    # ---- logging ----
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[http] " + (fmt % args) + "\n")


class ThreadingServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


# ===========================================================================
# Preflight
# ===========================================================================
def preflight() -> None:
    ok = True

    if not INDEX_HTML.exists():
        print(f"  ! missing required: {INDEX_HTML.name}")
        ok = False

    for f in ("example.vrm", "idle.vrma"):
        if not (ASSETS_DIR / f).exists():
            print(f"  ! missing required: assets/{f}")
            ok = False

    if not MOTION_TO_VRMA_BIN.exists():
        print(f"  ! missing binary: {MOTION_TO_VRMA_BIN.relative_to(ROOT)}")
        ok = False

    if ok:
        print("  ✓ index.html + assets + motion_to_vrma OK")

    for f in ("0.vrma", "1.vrma"):
        mark = "✓" if (ASSETS_DIR / f).exists() else "○"
        print(f"  {mark} assets/{f} (optional)")


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    print("SimpleLove streaming demo")
    print(f"root: {ROOT}")
    print(f"platform: {sys.platform}  (os.name={os.name})")
    preflight()
    print()
    start_script = "scripts\\start_servers.bat" if os.name == "nt" else "./scripts/start_servers.sh"
    print(f"NOTE: run `{start_script}` first (fused + t2m_infer).")
    print()

    with ThreadingServer(("", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}/"
        print(f"serving on {url}  (Ctrl-C to stop)")
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
