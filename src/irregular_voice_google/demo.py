"""Live booking demo: ask each question, record the answer, show what the app would do.

    uv run ivg-demo --profile data/speakers/<id>/profile.json \\
        --model models/ggml/<adapter>-q5_0.bin --say

For each booking question (destination, origin, date, airline) the demo
records one answer from the microphone (Enter to start, Enter to stop),
transcribes it with whisper.cpp (plain decoding by default: the adapter was
trained and scored without prompt or grammar, and the grammar made it mishear
clear answers; ``--prompt``/``--grammar`` turn them on) and runs ``questions.resolve``: accept, confirm ("Meinten Sie …?") or ask
again. It ends by reading back the booking for a final yes/no.

Every take is kept under ``data/demo/<timestamp>/`` (git-ignored) with its
transcript, so demo sessions can become training data later (with consent).
macOS only (``say``, ffmpeg's avfoundation input).

``--web`` runs the same dialog as a local page (http://localhost:8766, bound
to 127.0.0.1): the browser speaks the questions (German system voice) and
records with a big mic button; this machine transcribes and decides.
``--web --lan`` serves it over HTTPS with a token for a phone on the same
Wi-Fi; an Android emulator can use plain ``adb reverse tcp:8766 tcp:8766``
and http://localhost:8766 instead.

``--compare`` instead writes a local page (under ``--out``, git-ignored) with
the speaker's held-out test clips: waveform, spectrogram, playback and the base
model's transcript next to the adapter's, wrong words marked:

    uv run ivg-demo --compare --model models/ggml/<adapter>-q5_0.bin
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import subprocess
import random
import secrets
import threading
import time
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from irregular_voice_google import cpp, questions, snap
import jiwer

from irregular_voice_google.guard import guard
from irregular_voice_google.manifest import load_manifest
from irregular_voice_google.preprocess import SAMPLE_RATE, load
from irregular_voice_google.profile import Profile, load_profile
from irregular_voice_google.recorder import MAX_UPLOAD_BYTES, lan_ips, self_signed_context
from irregular_voice_google.text import normalize

FLOW = ["destination", "origin", "date", "airline"]
MAX_TRIES = 3


def list_mics() -> None:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                         capture_output=True, text=True).stderr
    print("\n".join(line.split("] ", 1)[-1] for line in out.splitlines() if "AVFoundation" in line))


def record(dest: Path, mic: str) -> None:
    input("  [Enter] zum Aufnehmen ")
    proc = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                             "-f", "avfoundation", "-i", f":{mic}", "-ac", "1", "-ar", str(SAMPLE_RATE),
                             "-c:a", "pcm_s16le", str(dest)], stdin=subprocess.PIPE)
    input("  ● Aufnahme läuft … [Enter] zum Beenden ")
    proc.communicate(b"q")


def _marked(hyp: str, ref: str) -> list[tuple[str, bool]]:
    """Hypothesis words, each flagged by whether it occurs in the reference."""
    ref_words = set(normalize(ref).split())
    return [(w, all(n in ref_words for n in normalize(w).split())) for w in hyp.split()]


def compare(args: argparse.Namespace) -> None:
    """Base model vs adapter on test clips the adapter never trained on, as a local web page."""
    clips = [u for u in load_manifest(args.manifest) if u.split == "test"]
    clips = random.Random(args.seed).sample(clips, min(args.n, len(clips))) if args.n else clips
    print(f"Transkribiere {len(clips)} Test-Aufnahmen mit beiden Modellen …")
    hyps = [cpp.transcribe(model, clips)[0] for model in (args.base, args.model)]
    names, model_files = ["Whisper (Basis)", "angepasst (LoRA)"], [Path(args.base).name, Path(args.model).name]
    if args.snap:  # adapter output with non-words snapped to real words
        snapper = snap.for_speaker(args.manifest)
        hyps.append([snapper.text(h) for h in hyps[1]])
        names.append("angepasst + Wortkorrektur")
        model_files.append(Path(args.model).name + " + snap.py")
    refs = [normalize(u.text) for u in clips]
    rows = []
    for i, u in enumerate(clips):
        wav = io.BytesIO()
        cpp.write_wav(load(u.audio), wav)
        rows.append({"ref": u.text, "wav": base64.b64encode(wav.getvalue()).decode(), "hyps": [
            {"words": _marked(h[i], u.text), "exact": normalize(h[i]) == refs[i],
             "wer": jiwer.wer(refs[i], normalize(h[i]) or "-")} for h in hyps]})
    data = {"models": names, "files": model_files, "clips": rows,
            "wer": [jiwer.wer(refs, [normalize(t) for t in h]) for h in hyps],
            "cer": [jiwer.cer(refs, [normalize(t) for t in h]) for h in hyps],
            "exact": [sum(normalize(t) == r for t, r in zip(h, refs)) for h in hyps]}
    page = files("irregular_voice_google").joinpath("compare.html").read_text(encoding="utf-8")
    out = Path(args.out) / f"compare-{datetime.now():%Y%m%d-%H%M%S}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False)), encoding="utf-8")
    for name, file, wer, cer in zip(data["models"], data["files"], data["wer"], data["cer"]):
        print(f"{name}: WER {wer:.1%}, CER {cer:.1%}  ({file})")
    print(out)
    subprocess.run(["open", str(out)])


YES = {"ja", "richtig", "ja, richtig", "bestätigen"}
NUMBERS = {"eins": 0, "erste": 0, "ersten": 0, "zwei": 1, "zwo": 1, "zweite": 1, "zweiten": 1,
           "drei": 2, "dritte": 2, "dritten": 2}
NO = ["nein", "falsch", "nein, falsch", "keins", "keine", "keins davon", "abbrechen", "wiederholen"]
ORDINAL = ["eins", "zwei", "drei"]


def is_yes(d: questions.Decision) -> bool:
    return d.action in ("accept", "confirm") and d.value in YES


class Dialog:
    """The booking conversation as a state machine, shared by the terminal and the web page.

    Each step says what to speak and which question the next answer belongs to
    (``listen``; None once the dialog is done). ``answer`` takes the decision
    for that answer: accept moves on; confirm or choose offers one to three
    ``choices`` ("Meinten Sie eins: Berlin, zwei: Bern, oder drei: Bremen?"),
    picked by naming one, its number, ja (for a single one) or a tap
    (``choose``); repeat asks again. After ``MAX_TRIES`` a question is left empty.
    """

    def __init__(self, qs: dict[str, questions.Question]):
        self.questions = qs
        self.booking: dict[str, str | None] = dict.fromkeys(FLOW)
        self.i, self.tries, self.phase = 0, 0, "ask"  # ask | confirm | final | done
        self.choices: list[str] = []
        self.confirmed: bool | None = None

    def question(self, name: str) -> questions.Question:
        """The question an answer is heard against; "choice" is ja/nein, the numbers and the offered values."""
        if name != "choice":
            return self.questions[name]
        values = [*YES, *NO, *NUMBERS, *self.choices]
        return questions.Question("choice", "", list(dict.fromkeys(values)), after=["bitte"])

    def step(self, say: str) -> dict:
        listen = {"ask": FLOW[self.i] if self.i < len(FLOW) else None, "confirm": "choice",
                  "final": "choice", "done": None}[self.phase]
        asking = FLOW[self.i] if self.phase in ("ask", "confirm") else None
        return {"say": say, "listen": listen, "asking": asking, "phase": self.phase, "booking": dict(self.booking),
                "choices": list(self.choices), "confirmed": self.confirmed}

    def start(self) -> dict:
        return self.step(self.questions[FLOW[0]].ask)

    def summary(self) -> str:
        b = {k: v or "?" for k, v in self.booking.items()}
        return f"Ein Flug von {b['origin']} nach {b['destination']}, am {b['date']}, mit {b['airline']}."

    def offer(self, choices: list[str]) -> dict:
        self.phase, self.choices = "confirm", choices[:3]
        if len(self.choices) == 1:
            return self.step(f"Meinten Sie {self.choices[0]}?")
        named = [f"{n}: {c}" for n, c in zip(ORDINAL, self.choices)]
        return self.step(f"Meinten Sie {', '.join(named[:-1])}, oder {named[-1]}?")

    def picked(self, d: questions.Decision) -> str | None:
        """The offered value an answer in the confirm phase picks, if any."""
        if d.action not in ("accept", "confirm"):
            return None
        if d.value in self.choices:
            return d.value
        if d.value in NUMBERS and NUMBERS[d.value] < len(self.choices):
            return self.choices[NUMBERS[d.value]]
        if d.value in YES and len(self.choices) == 1:
            return self.choices[0]
        return None

    def answer(self, d: questions.Decision) -> dict:
        if self.phase == "done":
            raise ValueError("dialog is done")
        if self.phase == "final":
            return self._finish(is_yes(d))
        if self.phase == "ask" and d.action == "accept":
            return self._next(d.value)
        if self.phase == "ask" and d.action in ("confirm", "choose"):
            return self.offer(d.candidates if d.action == "choose" else [d.value])
        if self.phase == "confirm" and (value := self.picked(d)):
            return self._next(value)
        return self._retry()

    def choose(self, value: str | None) -> dict:
        """A tapped button: one of ``choices`` (or "ja" in the final phase); None is "none of these"."""
        if self.phase == "final":
            return self._finish(value in YES)
        if self.phase == "confirm" and value in self.choices:
            return self._next(value)
        if self.phase == "confirm" and value is None:
            return self._retry()
        raise ValueError(f"nothing to choose in phase {self.phase}")

    def _finish(self, yes: bool) -> dict:
        self.confirmed, self.phase, self.choices = yes, "done", []
        return self.step("Vielen Dank, Ihr Flug ist gebucht." if yes else "Gut, dann fangen wir noch einmal an.")

    def _retry(self) -> dict:
        self.phase, self.choices, self.tries = "ask", [], self.tries + 1
        if self.tries < MAX_TRIES:
            return self.step("Entschuldigung, bitte noch einmal.")
        return self._next(None, "Entschuldigung, das überspringen wir. ")

    def _next(self, value: str | None, prefix: str = "") -> dict:
        self.booking[FLOW[self.i]] = value
        self.i, self.tries, self.choices, self.phase = self.i + 1, 0, [], "ask"
        if self.i < len(FLOW):
            return self.step(prefix + self.questions[FLOW[self.i]].ask)
        self.phase = "final"
        return self.step(prefix + self.summary() + " Ist das richtig?")


class Transcriber:
    """Transcribes answers with whisper.cpp and logs every take under ``data/demo/<timestamp>/``."""

    def __init__(self, args: argparse.Namespace, profile: Profile):
        self.args, self.profile = args, profile
        self.out = Path(args.out) / f"{datetime.now():%Y%m%d-%H%M%S}"
        self.out.mkdir(parents=True, exist_ok=True)
        self.takes: list[dict] = []

    def path(self, q: questions.Question) -> Path:
        return self.out / f"{len(self.takes) + 1:03d}-{q.name}.wav"

    def hear(self, wav: Path, q: questions.Question) -> questions.Decision:
        audio = load(wav, self.profile.preprocess)
        seconds = len(audio) / SAMPLE_RATE
        start = time.perf_counter()
        processed = self.out / "last.wav"
        cpp.write_wav(audio, processed)
        prompt = questions.prompt(q, self.profile) if self.args.prompt else ""
        grammar = questions.grammar(q) if self.args.grammar else None
        result = cpp.run(self.args.model, [processed], prompt, grammar)[0]
        text, flag = guard(result.text, seconds)
        decision = questions.resolve(q, text, result.min_p, self.args.threshold, bool(flag))
        if q.day_numbers and decision.value and (day := re.search(rf"(\d{{1,2}})\.?\s+{re.escape(decision.value)}",
                                                                     text, re.IGNORECASE)):
            decision.value = f"{day[1]}. {decision.value}"  # keep the day for the readback
        print(f"  gehört: {text!r}  (min_p {result.min_p:.2f}, {time.perf_counter() - start:.1f}s)"
              f"  -> {decision.action.upper()}" + (f" {decision.value}" if decision.value else "")
              + (f"  [guard: {flag}]" if flag else ""), flush=True)
        self.takes.append({"audio": wav.name, "question": q.name, "text": text, "min_p": round(result.min_p, 3),
                           "action": decision.action, "value": decision.value, "guard": flag})
        return decision

    def save(self, dialog: Dialog) -> None:
        (self.out / "session.json").write_text(json.dumps(
            {"model": str(self.args.model), "booking": dialog.booking, "confirmed": dialog.confirmed,
             "takes": self.takes}, indent=2, ensure_ascii=False), encoding="utf-8")
        (self.out / "last.wav").unlink(missing_ok=True)


class Session:
    """The dialog in the terminal: macOS ``say`` and the microphone via ffmpeg."""

    def __init__(self, args: argparse.Namespace, profile: Profile):
        self.args = args
        self.ear = Transcriber(args, profile)
        self.dialog = Dialog(questions.load_questions())

    def say(self, text: str) -> None:
        print(f"\n» {text}")
        if self.args.say:
            subprocess.run(["say", "-v", self.args.voice, text])

    def listen(self, q: questions.Question) -> questions.Decision:
        wav = self.ear.path(q)
        if self.args.wav is not None:  # replay files instead of the microphone (testing)
            if not self.args.wav:
                raise SystemExit("keine --wav Dateien mehr")
            wav = Path(self.args.wav.pop(0))
        else:
            record(wav, self.args.mic)
        return self.ear.hear(wav, q)

    def run(self) -> None:
        step = self.dialog.start()
        while True:
            self.say(step["say"])
            if step["listen"] is None:
                break
            step = self.dialog.answer(self.listen(self.dialog.question(step["listen"])))
        self.ear.save(self.dialog)
        print(f"\nAufnahmen und Protokoll: {self.ear.out}")


def make_handler(args: argparse.Namespace, profile: Profile, token: str | None = None):
    """Local web demo: the page speaks (browser TTS) and records; this side transcribes and decides."""
    page = files("irregular_voice_google").joinpath("demo.html").read_bytes()
    qs = questions.load_questions()
    lock = threading.Lock()  # one conversation at a time
    state: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def _json(self, obj, status=HTTPStatus.OK):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            if token is None or parse_qs(urlparse(self.path).query).get("token", [None])[0] == token:
                return True
            self.send_error(HTTPStatus.FORBIDDEN)
            return False

        def do_GET(self):
            if not self._authorized():
                return None
            if urlparse(self.path).path != "/":
                return self.send_error(HTTPStatus.NOT_FOUND)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def do_POST(self):
            if not self._authorized():
                return None
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if 0 < length <= MAX_UPLOAD_BYTES else b""
            with lock:
                if path == "/api/start":
                    state["ear"], state["dialog"] = Transcriber(args, profile), Dialog(qs)
                    return self._json({"step": state["dialog"].start()})
                if path not in ("/api/answer", "/api/choose"):
                    return self.send_error(HTTPStatus.NOT_FOUND)
                dialog = state.get("dialog")
                if dialog is None or dialog.phase == "done":
                    return self._json({"error": "kein Gespräch aktiv"}, HTTPStatus.CONFLICT)
                ear = state["ear"]
                if path == "/api/choose":  # a tapped option: {"value": "Berlin"}, "ja", or null for none
                    try:
                        value = json.loads(body or b"{}").get("value")
                        step = dialog.choose(value)
                    except (ValueError, AttributeError) as e:
                        return self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
                    ear.takes.append({"question": "choice", "tapped": value})
                    print(f"  getippt: {value!r}", flush=True)
                    if step["listen"] is None:
                        ear.save(dialog)
                    return self._json({"step": step})
                if body[:4] != b"RIFF" or body[8:12] != b"WAVE":
                    return self._json({"error": "keine WAV-Aufnahme"}, HTTPStatus.BAD_REQUEST)
                q = dialog.question(dialog.step("")["listen"])
                wav = ear.path(q)
                wav.write_bytes(body)
                try:
                    decision = ear.hear(wav, q)
                except Exception as e:  # keep the page usable; the take stays on disk
                    print(f"  Fehler: {e!r}", flush=True)
                    return self._json({"error": "Transkription fehlgeschlagen"}, HTTPStatus.INTERNAL_SERVER_ERROR)
                step = dialog.answer(decision)
                if step["listen"] is None:
                    ear.save(dialog)
                return self._json({"heard": ear.takes[-1], "step": step})

        def log_message(self, format, *args):
            pass

    return Handler


def serve(args: argparse.Namespace, profile: Profile) -> None:
    token = secrets.token_urlsafe(12) if args.lan else None
    server = ThreadingHTTPServer(("0.0.0.0" if args.lan else "127.0.0.1", args.port),
                                 make_handler(args, profile, token))
    if args.lan:  # a phone needs HTTPS for the microphone; the token keeps others on the network out
        server.socket = self_signed_context(Path("data/.recorder-cert")).wrap_socket(server.socket, server_side=True)
        print("Handy im selben WLAN: eine dieser Adressen öffnen (meist en0), Zertifikatswarnung akzeptieren.")
        for entry in lan_ips():
            ip, iface = entry.split(" ", 1)
            print(f"  https://{ip}:{args.port}/?token={token}  {iface}")
        url = f"https://localhost:{args.port}/?token={token}"
    else:
        url = f"http://localhost:{args.port}/"
    print(f"Web-Demo: {url}  (Strg+C beendet)", flush=True)
    subprocess.run(["open", url])
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", help="speaker profile (preprocessing steps and prompt phrases)")
    parser.add_argument("--model", help="ggml model from ivg-ggml (the speaker's merged adapter)")
    parser.add_argument("--mic", default="0", help="avfoundation audio device index (see --list-mics)")
    parser.add_argument("--list-mics", action="store_true")
    parser.add_argument("--say", action="store_true", help="speak the questions aloud")
    parser.add_argument("--voice", default="Anna", help="macOS German voice for --say")
    parser.add_argument("--threshold", type=float, default=0.5, help="min_p needed to accept without confirming")
    parser.add_argument("--prompt", action="store_true", help="per-question prompt (speaker phrases, values)")
    parser.add_argument("--grammar", action="store_true", help="per-question GBNF grammar (hurt the adapter)")
    parser.add_argument("--wav", nargs="+", help="use these files as the answers, in order (testing)")
    parser.add_argument("--out", default="data/demo")
    parser.add_argument("--compare", action="store_true", help="test clips page: base model vs adapter")
    parser.add_argument("--base", default="models/ggml/primeline__whisper-large-v3-turbo-german-q5_0.bin",
                        help="model to compare against (the adapter's base)")
    parser.add_argument("--manifest", default="data/processed/trim/manifest.csv", help="clips for --compare")
    parser.add_argument("--n", type=int, help="--compare only N random test clips")
    parser.add_argument("--snap", action="store_true", help="--compare: add the adapter with non-words snapped")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--web", action="store_true", help="local web page: mic button, spoken questions")
    parser.add_argument("--port", type=int, default=8766, help="--web port (bound to 127.0.0.1)")
    parser.add_argument("--lan", action="store_true", help="--web over HTTPS on the local network, for a phone")
    args = parser.parse_args()
    if args.list_mics:
        return list_mics()
    if not args.model or not Path(args.model).exists():
        parser.error("--model must be an existing ggml .bin (see ivg-ggml)")
    if not cpp.available():
        parser.error("whisper-cli not found (brew install whisper-cpp)")
    if args.compare:
        return compare(args)
    profile = load_profile(args.profile) if args.profile else Profile(speaker="demo")
    if args.web:
        return serve(args, profile)
    Session(args, profile).run()
