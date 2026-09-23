"""Local recording page for collecting prompted utterances.

Serves one prompt at a time; each saved take is written as a mono 16-bit WAV
plus a same-named ``.txt`` transcript under ``data/raw/<speaker>/<date>/`` —
the layout ``ivg-manifest`` reads. Audio never leaves this machine.

Browsers only allow microphone access on ``localhost`` or over HTTPS, so
``--lan`` (for recording on a phone) serves HTTPS with a self-signed
certificate and protects the page with a random token.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import socket
import ssl
import subprocess
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlparse

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def load_prompts(path: str | Path) -> list[dict]:
    prompts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            prompts.append({"id": hashlib.sha1(line.encode()).hexdigest()[:8], "text": line})
    return prompts


class Recorder:
    def __init__(self, prompts: list[dict], out_dir: Path, speaker: str, target: int):
        self.prompts = {p["id"]: p for p in prompts}
        self.order = [p["id"] for p in prompts]
        self.speaker_dir = out_dir / speaker
        self.target = target

    def counts(self) -> dict[str, int]:
        counts = dict.fromkeys(self.order, 0)
        for wav in self.speaker_dir.glob("*/p*_*.wav"):
            pid = wav.stem.split("_", 1)[0][1:]
            if pid in counts:
                counts[pid] += 1
        return counts

    def state(self) -> dict:
        counts = self.counts()
        # Fewest takes first, then prompt-file order.
        queue = sorted(self.order, key=lambda pid: (counts[pid], self.order.index(pid)))
        return {
            "target": self.target,
            "prompts": [{**self.prompts[pid], "count": counts[pid]} for pid in queue],
        }

    def save(self, pid: str, wav: bytes) -> Path:
        if pid not in self.prompts:
            raise KeyError(pid)
        if wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
            raise ValueError("not a WAV file")
        day_dir = self.speaker_dir / datetime.now().strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        stem = f"p{pid}_{datetime.now().strftime('%H%M%S%f')}"
        (day_dir / f"{stem}.wav").write_bytes(wav)
        (day_dir / f"{stem}.txt").write_text(self.prompts[pid]["text"] + "\n", encoding="utf-8")
        return day_dir / f"{stem}.wav"


def make_handler(recorder: Recorder, token: str | None):
    page = files("irregular_voice_google").joinpath("recorder.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def _authorized(self, query: dict) -> bool:
            if token is None or query.get("token", [None])[0] == token:
                return True
            self.send_error(HTTPStatus.FORBIDDEN)
            return False

        def _send(self, body: bytes, content_type: str, status=HTTPStatus.OK):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status=HTTPStatus.OK):
            self._send(json.dumps(obj, ensure_ascii=False).encode(), "application/json", status)

        def do_GET(self):
            url = urlparse(self.path)
            if not self._authorized(parse_qs(url.query)):
                return
            if url.path == "/":
                self._send(page, "text/html; charset=utf-8")
            elif url.path == "/api/state":
                self._json(recorder.state())
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if not self._authorized(query):
                return
            if url.path != "/api/record":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= MAX_UPLOAD_BYTES:
                self._json({"error": "bad length"}, HTTPStatus.BAD_REQUEST)
                return
            try:
                path = recorder.save(query.get("prompt", [""])[0], self.rfile.read(length))
            except (KeyError, ValueError) as e:
                self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
                return
            print(f"saved {path}", flush=True)
            self._json(recorder.state())

        def log_message(self, format, *args):
            pass

    return Handler


def self_signed_context(cert_dir: Path) -> ssl.SSLContext:
    cert, key = cert_dir / "cert.pem", cert_dir / "key.pem"
    if not cert.exists():
        cert_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "365",
             "-subj", "/CN=ivg-recorder", "-keyout", str(key), "-out", str(cert)],
            check=True, capture_output=True,
        )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    return context


def lan_ips() -> list[str]:
    """Non-loopback IPv4 addresses, labelled by interface (en0 is Wi-Fi on a Mac)."""
    try:
        out = subprocess.run(["ifconfig"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return [s.getsockname()[0]]
    ips, iface = [], ""
    for line in out.splitlines():
        if line and not line[0].isspace():
            iface = line.split(":", 1)[0]
        elif line.strip().startswith("inet "):
            ip = line.split()[1]
            if not ip.startswith("127."):
                ips.append(f"{ip} ({iface})")
    return ips


def main() -> None:
    parser = argparse.ArgumentParser(description="Record prompted utterances in the browser.")
    parser.add_argument("--prompts", default="resources/prompts/booking_de.txt")
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--speaker", default="patient")
    parser.add_argument("--target", type=int, default=2, help="Takes wanted per prompt.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--lan", action="store_true", help="Serve HTTPS on the local network for a phone.")
    args = parser.parse_args()

    recorder = Recorder(load_prompts(args.prompts), Path(args.out), args.speaker, args.target)
    token = secrets.token_urlsafe(12) if args.lan else None
    server = ThreadingHTTPServer(("0.0.0.0" if args.lan else "127.0.0.1", args.port), make_handler(recorder, token))

    if args.lan:
        server.socket = self_signed_context(Path("data/.recorder-cert")).wrap_socket(server.socket, server_side=True)
        print("Phone on the same Wi-Fi: open one of these (usually en0) and accept the certificate warning.")
        for entry in lan_ips():
            ip, iface = entry.split(" ", 1)
            print(f"  https://{ip}:{args.port}/?token={token}  {iface}")
        url = f"https://localhost:{args.port}/?token={token}"
    else:
        url = f"http://localhost:{args.port}/"
    print(f"Recording for '{args.speaker}' into {Path(args.out) / args.speaker}\n{url}\nCtrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
