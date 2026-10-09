"""Synthetic booking utterances in the user's own cloned voice (NeuTTS), as extra training data.

Same sentences as ``ivg-synth`` (answers to the booking questions plus whole requests), rendered not with generic
macOS voices but with an on-device TTS model (NeuTTS, Neuphonic) cloned from a few seconds of the user's own
speech. The reference clip and its transcript are taken from the manifest's *train* split, so the clone never
hears a dev/test sentence; sentences that appear in dev/test are skipped as in ``ivg-synth``.

    uv run ivg-synth-neutts --manifest data/processed/trim/manifest.csv --exclude data/processed/trim/manifest.csv \\
        --out data/synthetic/booking_neutts

The experiment this feeds (docs/open_questions.md, 2026-10-09): does a cloned-voice synthetic set lower the
adapter's held-out WER more than generic-voice synthetic data, or does a codec clone of a dysarthric voice teach
the adapter a voice that does not exist. Requirements: ``pip install neutts`` (and acceptance of the German
model's gated licence on Hugging Face by the account that downloads it); never on the Mac if it pulls torch.
Output is 24 kHz from the model, resampled here to 16 kHz mono 16-bit WAV so training runs unchanged.
"""

from __future__ import annotations

import argparse
import csv
import random
import wave
from pathlib import Path

import numpy as np

from irregular_voice_google.manifest import Utterance, load_manifest
from irregular_voice_google.preprocess import SAMPLE_RATE
from irregular_voice_google.synth import sentences
from irregular_voice_google.text import normalize

DEFAULT_BACKBONE = "neuphonic/neutts-nano-german"
DEFAULT_CODEC = "neuphonic/neucodec"
MODEL_SR = 24000
REF_MIN_S, REF_MAX_S = 4.0, 14.0        # the model wants 3-15 s; stay inside with a margin


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def pick_reference(utts: list[Utterance], manifest_dir: Path, rng: random.Random) -> Utterance:
    """A train-split utterance of 4-14 s, the longest within range (more voice to clone from); ties by seed."""
    cands = []
    for u in utts:
        if u.split != "train":
            continue
        p = manifest_dir / u.audio
        if p.suffix.lower() != ".wav" or not p.exists():
            continue
        s = wav_seconds(p)
        if REF_MIN_S <= s <= REF_MAX_S:
            cands.append((s, u))
    if not cands:
        raise SystemExit(f"no train-split wav between {REF_MIN_S} and {REF_MAX_S} s to clone from")
    best = max(s for s, _ in cands)
    return rng.choice([u for s, u in cands if s == best])


def resample(audio: np.ndarray, sr_in: int, sr_out: int = SAMPLE_RATE) -> np.ndarray:
    """Linear resampling (no new dependency); the clone's bandwidth is far under 8 kHz anyway."""
    if sr_in == sr_out:
        return audio.astype(np.float32)
    n = max(1, round(len(audio) * sr_out / sr_in))
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())


class NeuTTSRenderer:
    """Thin wrapper so tests can substitute a fake; holds the encoded reference."""

    def __init__(self, backbone: str, codec: str, device: str, ref_audio: Path, ref_text: str):
        from neutts import NeuTTS  # imported here: the Mac never installs it
        self.tts = NeuTTS(backbone_repo=backbone, backbone_device=device, codec_repo=codec, codec_device=device)
        self.ref_codes = self.tts.encode_reference(str(ref_audio))
        self.ref_text = ref_text

    def __call__(self, text: str) -> tuple[np.ndarray, int]:
        return np.asarray(self.tts.infer(text, self.ref_codes, self.ref_text), dtype=np.float32), MODEL_SR


def build(texts: list[str], render, out: Path, speaker: str, ref: Utterance) -> list[dict]:
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, text in enumerate(texts):
        dest = out / "audio" / f"{i:04d}.wav"
        if not dest.exists():
            audio, sr = render(text)
            write_wav(dest, resample(audio, sr))
        rows.append({"audio": f"audio/{dest.name}", "text": text, "split": "train", "speaker": speaker})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(texts)}", flush=True)
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio", "text", "split", "speaker"])
        writer.writeheader(); writer.writerows(rows)
    (out / "reference.txt").write_text(f"{ref.audio}\n{ref.text}\n", encoding="utf-8")
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True, help="the user's manifest; the reference clip comes from its train split")
    parser.add_argument("--exclude", action="append", default=[], help="manifest whose dev/test texts are skipped")
    parser.add_argument("--n", type=int, default=300, help="number of distinct sentences")
    parser.add_argument("--out", default="data/synthetic/booking_neutts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--codec", default=DEFAULT_CODEC)
    parser.add_argument("--device", default="cuda", help="cuda, cpu or mps")
    parser.add_argument("--speaker", default="neutts-clone")
    args = parser.parse_args(argv)

    utts = load_manifest(args.manifest)
    manifest_dir = Path(args.manifest).parent
    held_out = {normalize(u.text) for m in args.exclude for u in load_manifest(m) if u.split != "train"}
    texts = [t for t in sentences(args.n, args.seed) if normalize(t) not in held_out]
    ref = pick_reference(utts, manifest_dir, random.Random(args.seed))
    print(f"reference: {ref.audio} ({wav_seconds(manifest_dir / ref.audio):.1f} s, train split)")
    render = NeuTTSRenderer(args.backbone, args.codec, args.device, manifest_dir / ref.audio, ref.text)
    rows = build(texts, render, Path(args.out), args.speaker, ref)
    print(f"wrote {len(rows)} utterances to {Path(args.out) / 'manifest.csv'}")


if __name__ == "__main__":
    main()
