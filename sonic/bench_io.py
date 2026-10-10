"""Test-split I/O shared by the benches (no engine imports, so every venv can use it)."""

from __future__ import annotations

import csv
import wave
from pathlib import Path

import numpy as np


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as f:
        return np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(np.float32) / 32768


def test_clips(manifest: Path, split: str = "test") -> list[dict]:
    with manifest.open(encoding="utf-8") as f:
        return [{"audio": str(manifest.parent / r["audio"]), "reference": r["text"]}
                for r in csv.DictReader(f) if r["split"] == split]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
