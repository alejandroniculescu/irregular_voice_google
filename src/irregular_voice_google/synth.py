"""Synthetic booking utterances: German TTS, slowed and muffled, as extra training data.

The patient recordings contain no cities, airlines or dates, so the adapter
never hears him say them. This generates the answers to the booking questions
(``resources/questions_de.json``) plus whole booking requests with macOS
``say`` German voices, slowed down and low-passed, and writes a train-only
manifest for ``ivg-train --extra-manifest``:

    uv run ivg-synth --exclude data/processed/trim/manifest.csv

Sentences that also appear in the patient's dev/test split are skipped, so the
synthetic data cannot leak into evaluation. macOS only (``say``); the WAVs are
plain 16 kHz so training can run elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
import tempfile
from pathlib import Path

from irregular_voice_google.manifest import load_manifest
from irregular_voice_google.questions import load_questions
from irregular_voice_google.preprocess import SAMPLE_RATE
from irregular_voice_google.text import normalize

MONTHS = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober",
          "November", "Dezember"]
EXTRAS = [
    "Ich möchte einen Flug buchen.", "Einen Hinflug bitte.", "Hin und Rückflug bitte.", "Nur einfach.",
    "Einen Direktflug bitte.", "Ich möchte nicht umsteigen.", "Einen Fensterplatz bitte.", "Einen Gangplatz bitte.",
    "Ich brauche einen Rollstuhl.", "Ich reise mit Begleitperson.", "Nur Handgepäck.", "Mit einem Koffer.",
    "Economy bitte.", "Business bitte.", "Was kostet der Flug?", "Ich möchte bezahlen.", "Vormittags bitte.",
    "Nachmittags bitte.", "Abends bitte.", "Für zwei Personen.", "Für eine Person.", "Bitte wiederholen.",
]


def _answer(q, rng: random.Random) -> str:
    value = rng.choice(q.values)
    if q.day_numbers and value in MONTHS and rng.random() < 0.7:
        value = f"{rng.randint(1, 31)}. {value}"
    words = [rng.choice(q.before)] if q.before and rng.random() < 0.7 else []
    words.append(value)
    if q.after and rng.random() < 0.3:
        words.append(rng.choice(q.after))
    text = " ".join(words)
    return text[:1].upper() + text[1:] + "."


def _request(qs, rng: random.Random) -> str:
    dest, origin = rng.sample(qs["destination"].values, 2)
    parts = [f"Ich möchte von {origin} nach {dest} fliegen"]
    if rng.random() < 0.5:
        parts.append(f"am {rng.randint(1, 28)}. {rng.choice(MONTHS)}")
    if rng.random() < 0.4:
        parts.append(f"mit {rng.choice(qs['airline'].values)}")
    return " ".join(parts) + "."


def sentences(n: int, seed: int = 0) -> list[str]:
    rng, qs = random.Random(seed), load_questions()
    out = set(EXTRAS)
    while len(out) < n:
        out.add(_request(qs, rng) if rng.random() < 0.25 else _answer(rng.choice(list(qs.values())), rng))
    return sorted(out)


def german_voices() -> list[str]:
    lines = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=True).stdout.splitlines()
    return [line.split("de_DE")[0].strip() for line in lines if " de_DE " in line]


def render(text: str, voice: str, rate: int, cutoff: int, dest: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        aiff = Path(tmp) / "say.aiff"
        subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(aiff), text], check=True)
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(aiff), "-af",
                        f"lowpass=f={cutoff},lowpass=f={cutoff}", "-ac", "1", "-ar", str(SAMPLE_RATE),
                        "-c:a", "pcm_s16le", str(dest)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=300, help="number of distinct sentences")
    parser.add_argument("--exclude", action="append", default=[], help="manifest whose dev/test texts are skipped")
    parser.add_argument("--out", default="data/synthetic/booking")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    held_out = {normalize(u.text) for m in args.exclude for u in load_manifest(m) if u.split != "train"}
    texts = [t for t in sentences(args.n, args.seed) if normalize(t) not in held_out]
    voices, rng = german_voices(), random.Random(args.seed)
    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, text in enumerate(texts):
        voice, rate, cutoff = rng.choice(voices), rng.randint(100, 150), rng.randint(700, 1500)
        dest = out / "audio" / f"{i:04d}.wav"
        if not dest.exists():
            render(text, voice, rate, cutoff, dest)
        rows.append({"audio": f"audio/{dest.name}", "text": text, "split": "train",
                     "speaker": "tts-" + voice.split()[0].lower()})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(texts)}", flush=True)
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio", "text", "split", "speaker"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} utterances ({len(voices)} voices) to {out / 'manifest.csv'}")
