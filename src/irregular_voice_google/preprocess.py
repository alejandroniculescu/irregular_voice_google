"""Audio preprocessing variants, materialised under ``data/processed/<variant>/``.

A variant is a comma-separated list of steps, applied in order with ffmpeg:

- ``trim``        cut leading/trailing silence (keeps 0.15 s either side)
- ``tempo=1.2``   speed up without changing pitch (dysarthric speech is slow)
- ``eq=6``        boost 2–5 kHz by N dB (tests the "muffled" hypothesis)

``ivg-preprocess --steps trim,tempo=1.2`` writes the processed audio plus a
manifest with the same texts and splits, so train/eval just take
``--manifest data/processed/<variant>/manifest.csv``. The same ``load()`` must
be used on live audio at inference time, or the model sees different input.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np

from irregular_voice_google.manifest import load_manifest

SAMPLE_RATE = 16_000
_TRIM = "silenceremove=start_periods=1:start_silence=0.15:start_threshold=-40dB"


def parse_steps(steps: str) -> list[tuple[str, float | None]]:
    parsed = []
    for step in filter(None, (s.strip() for s in steps.split(","))):
        name, _, value = step.partition("=")
        if name not in {"trim", "tempo", "eq"} or (name != "trim") != bool(value):
            raise ValueError(f"bad step {step!r}; expected trim, tempo=<x> or eq=<dB>")
        parsed.append((name, float(value) if value else None))
    return parsed


def ffmpeg_filters(steps: str) -> str:
    filters = []
    for name, value in parse_steps(steps):
        if name == "trim":
            filters += [_TRIM, "areverse", _TRIM, "areverse"]
        elif name == "tempo":
            if not 0.5 <= value <= 2.0:
                raise ValueError("tempo must be between 0.5 and 2.0")
            filters.append(f"atempo={value}")
        elif name == "eq":
            # Leave headroom first: the source audio already peaks at 0 dBFS.
            filters += [f"volume=-{value}dB", f"equalizer=f=3500:t=h:w=3000:g={value}"]
    return ",".join(filters) or "anull"


def variant_name(steps: str) -> str:
    return "_".join(f"{n}{'' if v is None else f'{v:g}'}" for n, v in parse_steps(steps)) or "raw"


def load(path: str | Path, steps: str = "") -> np.ndarray:
    """Decode any audio file to 16 kHz mono float32, applying the preprocessing steps."""
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path), "-af", ffmpeg_filters(steps),
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(out, np.float32)


def process_manifest(manifest: str | Path, steps: str, out_root: str | Path = "data/processed") -> Path:
    """Write processed WAVs and a manifest for one variant; returns the manifest path."""
    manifest = Path(manifest)
    out_dir = Path(out_root) / variant_name(steps)
    rows = []
    for u in load_manifest(manifest):
        rel = u.audio.relative_to(manifest.parent.resolve(), walk_up=True)
        dest = (out_dir / "audio" / Path(*(p for p in rel.parts if p != ".."))).with_suffix(".wav")
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(u.audio), "-af", ffmpeg_filters(steps),
                 "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dest)],
                check=True,
            )
        rows.append({"audio": dest.relative_to(out_dir).as_posix(), "text": u.text, "split": u.split,
                     "speaker": u.speaker})
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio", "text", "split", "speaker"])
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "variant.json").write_text(
        json.dumps({"steps": steps, "ffmpeg": ffmpeg_filters(steps), "source": str(manifest)}, indent=2),
        encoding="utf-8",
    )
    return out_dir / "manifest.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--steps", required=True, help='e.g. "trim", "trim,tempo=1.2,eq=6"')
    parser.add_argument("--out", default="data/processed")
    args = parser.parse_args()
    print(f"wrote {process_manifest(args.manifest, args.steps, args.out)}")
