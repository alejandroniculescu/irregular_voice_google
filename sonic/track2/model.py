"""SAPC2 Track 2 submission: streaming partials from one expert, the final from a router over several experts.

Streaming: every ``PARTIAL_EVERY`` seconds of new audio, the partial expert re-decodes the whole buffer (encoder
sized to the buffer); a word is shown only once two consecutive decodes agree on it (LocalAgreement-2), and nothing
is decoded before an energy gate has seen speech, so the first word cannot settle before it is spoken (reject
rule 1). When the decoder lags the audio by more than ``SLACK`` (queued chunks in real time), partial decodes are
shed until it has caught up; the final is never shed. The final decodes the audio trimmed to its last loud 10 ms
frame + ``PAD``, a function of the audio alone; once ``SPEC_AFTER`` of silence follows speech it is decoded
speculatively, and ``input_finished`` returns that result when no speech came after (TTLT ≈ 0). Mixture + routing: at ``input_finished`` the first expert decodes the full audio; if its mean token
log-prob is below ``SONIC_ESCALATE`` the other experts decode too and the most confident one is the final
(confidence cascade, S1 in docs/open_questions.md). The final depends on the audio alone, so Pass 1 and Pass 2 agree
(reject rule 2).

Configuration by environment, so the same file runs locally and in the submission:
``SONIC_MODELS`` (dir of the ggml experts), ``SONIC_EXPERTS`` (comma list of file stems, first one streams),
``SONIC_THREADS``, ``SONIC_LANG``, ``SONIC_SNAP`` (a manifest: its train transcripts, the commands and lexicons feed
``snap`` on the final, non-words to the closest same-sounding real word; empty = off; S4: WER 13.2 -> 9.1 %), ``SONIC_PARTIAL_EVERY`` (seconds), ``SONIC_PARTIAL_CTX`` and ``SONIC_FINAL_CTX``
(encoder window: ``auto`` sizes it to the audio with a floor of 512, ``0`` is the full 30 s, or a number of
positions; the final uses the full window: the soup loses 1.3 points at 1024, S3), ``SONIC_ESCALATE`` (log-prob threshold).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent.parent), str(HERE.parent.parent / "src")]   # repo root (sonic) and src (package)

from sonic.engine import SAMPLE_RATE, Expert  # noqa: E402
from sonic.route import cascade  # noqa: E402

MODELS = Path(os.environ.get("SONIC_MODELS", HERE.parent.parent / "models" / "ggml"))
EXPERTS = os.environ.get("SONIC_EXPERTS", "soup-dora-r16-synth-q8_0,models__lora-dora-r32-q5_0,models__lora-r16-q5_0,models__lora-aug-synth-q5_0").split(",")
THREADS = int(os.environ.get("SONIC_THREADS", "4"))
LANG = os.environ.get("SONIC_LANG", "de")
PARTIAL_EVERY = float(os.environ.get("SONIC_PARTIAL_EVERY", "0.5"))
PARTIAL_CTX = os.environ.get("SONIC_PARTIAL_CTX", "auto")
FINAL_CTX = os.environ.get("SONIC_FINAL_CTX", "0")
ESCALATE = float(os.environ.get("SONIC_ESCALATE", "-0.20"))     # S4: escalates ~5 % of clips; the soup holds the gain
SNAP = os.environ.get("SONIC_SNAP", str(HERE.parent.parent / "data" / "processed" / "trim" / "manifest.csv"))
SLACK = 0.2               # seconds the decoder may lag the audio before partial decodes are shed
FRAME = 160               # 10 ms loudness frames, at absolute offsets, for the trim
TRIM_RMS = 0.01           # a frame above this (about -40 dBFS) is speech for the trim
PAD = 0.2                 # seconds kept after the last loud frame
SPEC_AFTER = 0.3          # seconds of trailing silence before the final is decoded speculatively (>= PAD)
GATE_RMS = 0.01           # speech gate: 100 ms chunk RMS above this (about -40 dBFS)
GATE_CHUNKS = 2           # ...for this many chunks in a row


def loud_frames(audio: np.ndarray, start_frame: int, end_frame: int) -> np.ndarray:
    fr = audio[start_frame * FRAME: end_frame * FRAME].reshape(-1, FRAME)
    return np.sqrt((fr ** 2).mean(axis=1)) > TRIM_RMS


def trim_end(audio: np.ndarray) -> tuple[int, int]:
    """(key, end): key = sample after the last loud frame (0 if none); end = where the final's audio stops. A function
    of the audio alone, so Pass 1 and Pass 2 cut the same samples."""
    loud = np.nonzero(loud_frames(audio, 0, len(audio) // FRAME))[0]
    if not loud.size:
        return 0, len(audio)
    key = int(loud[-1] + 1) * FRAME
    return key, min(len(audio), key + int(PAD * SAMPLE_RATE))


def ctx_of(value: str) -> int | None:
    return None if value == "auto" else int(value)


def common_prefix(a: list[str], b: list[str]) -> list[str]:
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out


class Model:
    def __init__(self):
        self.experts = [Expert(str(MODELS / f"{stem}.bin"), stem, threads=THREADS, language=LANG) for stem in EXPERTS]
        self.snap = None
        if SNAP and Path(SNAP).exists():
            from irregular_voice_google import snap
            self.snap = snap.for_speaker(SNAP)
        self._partial_callback = lambda _text: None
        self.reset()

    def set_partial_callback(self, callback) -> None:
        self._partial_callback = callback

    def reset(self) -> None:
        self.chunks: list[np.ndarray] = []
        self.n = 0
        self.loud = 0
        self.speech = False
        self.last_decode_at = 0
        self.prev_words: list[str] = []
        self.committed: list[str] = []
        self.t0 = None
        self.frames_done = 0          # loudness frames examined so far
        self.last_loud_end = 0        # sample after the last loud frame
        self.spec_key, self.spec = -1, None

    def accept_chunk(self, audio_chunk: np.ndarray) -> str:
        chunk = np.asarray(audio_chunk, dtype=np.float32)
        if self.t0 is None:
            self.t0 = time.monotonic() - len(chunk) / SAMPLE_RATE
        self.chunks.append(chunk); self.n += len(chunk)
        self._track_loudness()
        behind = time.monotonic() - self.t0 > self.n / SAMPLE_RATE + SLACK   # backpressure: shed this partial
        if not self.speech:
            self.loud = self.loud + 1 if chunk.size and float(np.sqrt(np.mean(chunk ** 2))) > GATE_RMS else 0
            self.speech = self.loud >= GATE_CHUNKS
        if self.speech and not behind and self.n - self.last_decode_at >= PARTIAL_EVERY * SAMPLE_RATE:
            self.last_decode_at = self.n
            words = self.experts[0].decode(np.concatenate(self.chunks), ctx_of(PARTIAL_CTX)).text.split()
            agreed = common_prefix(self.prev_words, words)
            if len(agreed) > len(self.committed):
                self.committed = agreed
            self.prev_words = words
            self._partial_callback(" ".join(self.committed))
        if (self.speech and self.last_loud_end and self.spec_key != self.last_loud_end
                and self.n - self.last_loud_end >= SPEC_AFTER * SAMPLE_RATE):
            # trailing silence: decode the final now; input_finished reuses it if no speech follows
            key, end = self.last_loud_end, self.last_loud_end + int(PAD * SAMPLE_RATE)
            self.spec_key, self.spec = key, self._final(np.concatenate(self.chunks)[:end])
        return " ".join(self.committed)

    def _track_loudness(self) -> None:
        total = self.n // FRAME
        if total > self.frames_done:
            audio = np.concatenate(self.chunks)
            loud = np.nonzero(loud_frames(audio, self.frames_done, total))[0]
            if loud.size:
                self.last_loud_end = (self.frames_done + int(loud[-1]) + 1) * FRAME
            self.frames_done = total

    def _final(self, audio: np.ndarray) -> tuple[str, list[int]]:
        def decode(k: int) -> tuple[str, float]:
            d = self.experts[k].decode(audio, ctx_of(FINAL_CTX))
            return d.text, d.logprob
        text, ran = cascade(decode, len(self.experts), ESCALATE)
        return (self.snap.text(text) if self.snap else text), ran

    def input_finished(self) -> str:
        if not self.chunks:
            return ""
        audio = np.concatenate(self.chunks)
        key, end = trim_end(audio)
        self.reused = key == self.spec_key and self.spec is not None
        final, self.ran = self.spec if self.reused else self._final(audio[:end])
        self._partial_callback(final)
        return final
