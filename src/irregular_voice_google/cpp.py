"""whisper.cpp backend: run ``whisper-cli`` on a ggml model from ``ivg-ggml``.

This is the runtime the app will use (Metal on a Mac, small quantized models,
no Python). Compared with the HF pipeline it adds grammar-constrained
decoding and returns a probability for every token.

Install with ``brew install whisper-cpp``. The prompt is capped by whisper.cpp
at half the 448-token text context, as in the HF backend.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from irregular_voice_google.guard import guard
from irregular_voice_google.manifest import Utterance
from irregular_voice_google.preprocess import SAMPLE_RATE, load

WHISPER_CLI = "whisper-cli"


@dataclass
class Result:
    text: str
    tokens: list[tuple[str, float]]  # (token text, probability), special tokens dropped

    @property
    def min_p(self) -> float:
        return min((p for t, p in self.tokens if t.strip()), default=0.0)


def available() -> bool:
    return shutil.which(WHISPER_CLI) is not None


def write_wav(audio: np.ndarray, path: Path | BinaryIO) -> None:
    with wave.open(str(path) if isinstance(path, (str, Path)) else path, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def _parse(path: Path) -> Result:
    segments = json.loads(path.read_text(encoding="utf-8"))["transcription"]
    tokens = [(t["text"], t["p"]) for s in segments for t in s.get("tokens", []) if not t["text"].startswith("[_")]
    return Result("".join(s["text"] for s in segments).strip(), tokens)


def run(model: str | Path, wavs: list[Path], prompt: str = "", grammar: str | None = None,
        language: str = "de", beam_size: int = 5) -> list[Result]:
    """Transcribe 16 kHz WAVs in one ``whisper-cli`` call (the model loads once)."""
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [WHISPER_CLI, "-m", str(model), "-l", language, "-bs", str(beam_size), "-nt", "-np", "-ojf"]
        if prompt:
            cmd += ["--prompt", prompt]
        if grammar:
            (Path(tmp) / "answer.gbnf").write_text(grammar, encoding="utf-8")
            cmd += ["--grammar", str(Path(tmp) / "answer.gbnf"), "--grammar-rule", "root"]
        for i, wav in enumerate(wavs):
            cmd += ["-f", str(wav), "-of", str(Path(tmp) / str(i))]
        subprocess.run(cmd, check=True, capture_output=True)
        return [_parse(Path(tmp) / f"{i}.json") for i in range(len(wavs))]


def transcribe(model: str | Path, utterances: list[Utterance], prompt: str = "",
               grammar: str | None = None) -> tuple[list[str], list[str | None], list[float]]:
    """Guarded hypotheses, guard flags and each hypothesis's lowest token probability."""
    with tempfile.TemporaryDirectory() as tmp:
        wavs, seconds = [], []
        for i, utt in enumerate(utterances):
            audio = load(utt.audio)
            seconds.append(len(audio) / SAMPLE_RATE)
            wavs.append(Path(tmp) / f"{i}.wav")
            write_wav(audio, wavs[-1])
        results = run(model, wavs, prompt, grammar)
    hypotheses, flags = zip(*(guard(r.text, s) for r, s in zip(results, seconds))) if results else ((), ())
    for i, (h, f) in enumerate(zip(hypotheses, flags), 1):
        print(f"  [{i}/{len(utterances)}] {h}" + (f"  [guard: {f}]" if f else ""))
    return list(hypotheses), list(flags), [round(r.min_p, 3) for r in results]
