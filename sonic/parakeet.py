"""Parakeet (FastConformer transducer) as a CPU expert, and an offline bench on the test split (P1 in
docs/open_questions.md). One wrapper for both uses: ``nvidia/parakeet-tdt-0.6b-v3`` (German, for Christian) and
``dys-asr/parakeet-rnnt-0.6b-sapc12-syn`` (English, SAP-tuned, for Track 2).

    PYTHONPATH=src:. .venv-parakeet/bin/python -m sonic.parakeet --model nvidia/parakeet-tdt-0.6b-v3

Writes ``results/sonic/<timestamp>/<name>.csv`` (git-ignored; audio, reference, hypothesis, seconds) and prints
numbers only, never transcripts.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

SAMPLE_RATE = 16000


class Parakeet:
    def __init__(self, model_id: str, threads: int = 4):
        torch.set_num_threads(threads)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).eval()     # ParakeetForTDT / ParakeetForRNNT from config

    def decode(self, audio: np.ndarray) -> tuple[str, float]:
        """(text, seconds) for one 16 kHz float32 clip; greedy."""
        t = time.perf_counter()
        inputs = self.processor(audio.astype(np.float32), sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.inference_mode():
            out = self.model.generate(**inputs)
        seqs = out.sequences if hasattr(out, "sequences") else out
        text = self.processor.batch_decode(seqs, skip_special_tokens=True)[0].strip()
        return text, time.perf_counter() - t


@dataclass
class Decode:
    text: str
    logprob: float
    seconds: float


class ParakeetExpert(Parakeet):
    """The ``sonic.engine.Expert`` interface for the streaming model: ``decode(audio, ctx)`` (``ctx`` ignored: the
    FastConformer encoder always runs over exactly the audio it is given). ``logprob`` is the transducer
    log-likelihood of the decoded text given the audio (summed over alignments; the training loss, which is
    already per label token): a confidence for the router; computed only with ``confidence=True`` (one more forward pass)."""

    def __init__(self, model_id: str, threads: int = 4, confidence: bool = False):
        super().__init__(model_id, threads)
        self.name, self.confidence = model_id, confidence

    def decode(self, audio: np.ndarray, ctx: int | None = None) -> Decode:
        t = time.perf_counter()
        text, _ = super().decode(audio)
        lp = 0.0
        if self.confidence:
            lp = self.logprob(audio, text)
        return Decode(text, lp, time.perf_counter() - t)

    def logprob(self, audio: np.ndarray, text: str) -> float:
        if not text:
            return float("-inf")
        inputs = self.processor(audio.astype(np.float32), text=[text], sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.inference_mode():
            return -float(self.model(**inputs).loss)      # NeMo "mean" reduction: already per label token


def main(argv=None) -> None:
    from irregular_voice_google import snap
    from irregular_voice_google.per import gruut_g2p
    from sonic.bench_io import read_wav, test_clips, write_csv
    from sonic.route import fmt, scores

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3")
    parser.add_argument("--manifest", default="data/processed/trim/manifest.csv")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--split", default="test", help="manifest split to decode (dev for fitting routers)")
    args = parser.parse_args(argv)

    clips = test_clips(Path(args.manifest), args.split)
    expert = ParakeetExpert(args.model, threads=args.threads, confidence=True)
    expert.decode(np.zeros(SAMPLE_RATE, np.float32))                  # warm-up, not timed
    rows = []
    for c in clips:
        d = expert.decode(read_wav(Path(c["audio"])))
        rows.append({**c, "hypothesis": d.text, "logprob": round(d.logprob, 4), "seconds": round(d.seconds, 3)})
    out = Path("results/sonic") / datetime.now().strftime("%Y%m%d-%H%M%S"); out.mkdir(parents=True)
    name = args.model.rstrip("/").split("/")[-1] + ("" if args.split == "test" else f"@{args.split}")
    write_csv(out / f"{name}.csv", rows)
    g2p, sn = gruut_g2p(), snap.for_speaker(args.manifest)
    t = [r["seconds"] for r in rows]
    print(f"{name}        {fmt(scores(rows, g2p))}  sec/clip {np.mean(t):.2f} (max {np.max(t):.2f})")
    print(f"{name} +snap  {fmt(scores([{**r, 'hypothesis': sn.text(r['hypothesis'])} for r in rows], g2p))}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
