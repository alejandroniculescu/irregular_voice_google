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
import time
from datetime import datetime
from importlib.resources import files
from pathlib import Path

from irregular_voice_google import cpp, questions
import jiwer

from irregular_voice_google.guard import guard
from irregular_voice_google.manifest import load_manifest
from irregular_voice_google.preprocess import SAMPLE_RATE, load
from irregular_voice_google.profile import Profile, load_profile
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
    refs = [normalize(u.text) for u in clips]
    rows = []
    for i, u in enumerate(clips):
        wav = io.BytesIO()
        cpp.write_wav(load(u.audio), wav)
        rows.append({"ref": u.text, "wav": base64.b64encode(wav.getvalue()).decode(), "hyps": [
            {"words": _marked(h[i], u.text), "exact": normalize(h[i]) == refs[i],
             "wer": jiwer.wer(refs[i], normalize(h[i]) or "-")} for h in hyps]})
    data = {"models": ["Whisper (Basis)", "angepasst (LoRA)"], "files": [Path(args.base).name, Path(args.model).name],
            "clips": rows,
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


class Session:
    def __init__(self, args: argparse.Namespace, profile: Profile):
        self.args, self.profile = args, profile
        self.questions = questions.load_questions()
        self.out = Path(args.out) / f"{datetime.now():%Y%m%d-%H%M%S}"
        self.out.mkdir(parents=True, exist_ok=True)
        self.takes: list[dict] = []

    def say(self, text: str) -> None:
        print(f"\n» {text}")
        if self.args.say:
            subprocess.run(["say", "-v", self.args.voice, text])

    def listen(self, q: questions.Question) -> questions.Decision:
        wav = self.out / f"{len(self.takes) + 1:03d}-{q.name}.wav"
        if self.args.wav is not None:  # replay files instead of the microphone (testing)
            if not self.args.wav:
                raise SystemExit("keine --wav Dateien mehr")
            wav = Path(self.args.wav.pop(0))
        else:
            record(wav, self.args.mic)
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
              + (f"  [guard: {flag}]" if flag else ""))
        self.takes.append({"audio": wav.name, "question": q.name, "text": text, "min_p": round(result.min_p, 3),
                           "action": decision.action, "value": decision.value, "guard": flag})
        return decision

    def yes(self) -> bool | None:
        d = self.listen(self.questions["confirm"])
        if d.action == "repeat" or d.value is None:
            return None
        return d.value in {"ja", "richtig", "ja, richtig", "bestätigen"}

    def ask(self, name: str) -> str | None:
        q = self.questions[name]
        self.say(q.ask)
        for _ in range(MAX_TRIES):
            d = self.listen(q)
            if d.action == "accept":
                return d.value
            if d.action == "confirm":
                self.say(f"Meinten Sie {d.value}?")
                if self.yes():
                    return d.value
            self.say("Entschuldigung, bitte noch einmal.")
        return None

    def run(self) -> None:
        booking = {name: self.ask(name) for name in FLOW}
        summary = (f"Ein Flug von {booking['origin'] or '?'} nach {booking['destination'] or '?'}, "
                   f"am {booking['date'] or '?'}, mit {booking['airline'] or '?'}.")
        self.say(summary + " Ist das richtig?")
        confirmed = self.yes()
        self.say("Vielen Dank, Ihr Flug ist gebucht." if confirmed else "Gut, dann fangen wir noch einmal an.")
        (self.out / "session.json").write_text(json.dumps(
            {"model": str(self.args.model), "booking": booking, "confirmed": confirmed, "takes": self.takes},
            indent=2, ensure_ascii=False), encoding="utf-8")
        (self.out / "last.wav").unlink(missing_ok=True)
        print(f"\nAufnahmen und Protokoll: {self.out}")


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
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.list_mics:
        return list_mics()
    if not args.model or not Path(args.model).exists():
        parser.error("--model must be an existing ggml .bin (see ivg-ggml)")
    if not cpp.available():
        parser.error("whisper-cli not found (brew install whisper-cpp)")
    if args.compare:
        return compare(args)
    Session(args, load_profile(args.profile) if args.profile else Profile(speaker="demo")).run()
