"""Recording manifest: one CSV row per utterance.

Columns: ``audio`` (path relative to the manifest), ``text`` (reference
transcript), and optionally ``split`` and ``speaker``. Missing splits are
assigned deterministically from the transcript, so repeated recordings of the
same sentence always land in the same split and never leak from train to test.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from irregular_voice_google.text import normalize

SPLITS = ("train", "dev", "test")
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".webm", ".aac"}


@dataclass(frozen=True)
class Utterance:
    audio: Path
    text: str
    split: str
    speaker: str


def assign_split(text: str, test_frac: float = 0.2, dev_frac: float = 0.1) -> str:
    digest = hashlib.sha256(normalize(text).encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    if u < test_frac:
        return "test"
    if u < test_frac + dev_frac:
        return "dev"
    return "train"


def load_manifest(path: str | Path) -> list[Utterance]:
    path = Path(path)
    utterances = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = row["text"].strip()
            split = (row.get("split") or "").strip() or assign_split(text)
            if split not in SPLITS:
                raise ValueError(f"{path}: unknown split {split!r} for {row['audio']}")
            utterances.append(
                Utterance(
                    audio=(path.parent / row["audio"]).resolve(),
                    text=text,
                    split=split,
                    speaker=(row.get("speaker") or "").strip() or "patient",
                )
            )
    return utterances


def build_manifest(audio_dir: str | Path, out: str | Path, speaker: str = "patient") -> int:
    """Pair each audio file with a same-named ``.txt`` transcript and write a manifest.

    Audio without a transcript is skipped and reported; returns rows written.
    """
    audio_dir, out = Path(audio_dir), Path(out)
    rows, missing = [], []
    for audio in sorted(p for p in audio_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS):
        transcript = audio.with_suffix(".txt")
        if not transcript.exists():
            missing.append(audio)
            continue
        text = transcript.read_text(encoding="utf-8").strip()
        rel = Path(audio).resolve().relative_to(out.parent.resolve(), walk_up=True)
        rows.append({"audio": rel.as_posix(), "text": text, "split": assign_split(text), "speaker": speaker})

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio", "text", "split", "speaker"])
        writer.writeheader()
        writer.writerows(rows)
    for audio in missing:
        print(f"no transcript, skipped: {audio}")
    return len(rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=build_manifest.__doc__)
    parser.add_argument("audio_dir")
    parser.add_argument("--out", default="data/manifest.csv")
    parser.add_argument("--speaker", default="patient")
    args = parser.parse_args()
    n = build_manifest(args.audio_dir, args.out, args.speaker)
    print(f"wrote {n} rows to {args.out}")
