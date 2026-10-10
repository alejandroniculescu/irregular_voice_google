"""In-process whisper.cpp experts for the streaming build: each model loads once, the encoder runs only over the
audio it is given (``audio_ctx`` sized to the buffer instead of a padded 30 s), decoding is greedy with a token cap
(a short ``audio_ctx`` makes a model fine-tuned on padded audio loop), and every decode returns its mean token
log-probability for the confidence router.

CPU only: the hackathon scores on CPU, so ``use_gpu`` is off even on a Mac.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import _pywhispercpp as pw
import numpy as np
from pywhispercpp.model import Model

SAMPLE_RATE = 16000
FRAMES_PER_SECOND = 50          # encoder positions per second of audio (1500 for 30 s)
PAD_TO = 30 * SAMPLE_RATE      # whisper's window; see Expert.decode
TOKENS_PER_SECOND = 8           # cap: German speech is about 3-4 tokens/s, so 8 leaves room without allowing loops


@dataclass
class Decode:
    text: str
    logprob: float              # mean log-probability of the text tokens; -inf when nothing was decoded
    seconds: float


MIN_CTX = 512                   # below this the fine-tuned experts loop or invent (S1: PER ~100 % at audio + 1 s)


def audio_ctx_for(n_samples: int, margin_s: float = 1.0, quantum: int = 64, floor: int = MIN_CTX) -> int:
    """Encoder positions for this much audio plus a margin, rounded up to a multiple of ``quantum`` and at least
    ``floor``; 0 (= the full 30 s window) when that would not be shorter."""
    need = math.ceil((n_samples / SAMPLE_RATE + margin_s) * FRAMES_PER_SECOND)
    ctx = max(floor, -(-need // quantum) * quantum)
    return 0 if ctx >= 1500 else ctx


class Expert:
    """One merged adapter as a whisper.cpp model."""

    def __init__(self, path: str, name: str | None = None, threads: int = 4, language: str = "de"):
        self.name = name or path
        self.model = Model(path, context_params={"use_gpu": False}, redirect_whispercpp_logs_to=None,
                           n_threads=threads, language=language, print_progress=False, print_realtime=False,
                           no_context=True, single_segment=True, no_timestamps=True, suppress_blank=True,
                           temperature=0.0, temperature_inc=0.0)

    def decode(self, audio: np.ndarray, ctx: int | None = None) -> Decode:
        """``ctx`` None sizes the encoder to the audio; 0 is the full 30 s window; any other value is used as is."""
        import time
        t = time.perf_counter()
        ctx = audio_ctx_for(len(audio)) if ctx is None else ctx
        max_tokens = max(8, math.ceil(len(audio) / SAMPLE_RATE * TOKENS_PER_SECOND))
        # zero-pad to 30 s ourselves: neutral (D1: identical output padded or not) and kept as a guard. It was added
        # for run-to-run jitter on ahms (S5a) that D1 traced to the faulty CPU 0, not to whisper.cpp
        audio = np.pad(audio.astype(np.float32), (0, max(0, PAD_TO - len(audio))))
        segs = self.model.transcribe(audio, audio_ctx=ctx, max_tokens=max_tokens)
        text = " ".join(s.text.strip() for s in segs).strip()
        ctx_ = self.model._ctx
        eot = pw.whisper_token_eot(ctx_)
        lps = []
        for i in range(pw.whisper_full_n_segments(ctx_)):
            for j in range(pw.whisper_full_n_tokens(ctx_, i)):
                if pw.whisper_full_get_token_id(ctx_, i, j) >= eot:     # special tokens
                    continue
                lps.append(math.log(max(pw.whisper_full_get_token_p(ctx_, i, j), 1e-12)))
        return Decode(text, sum(lps) / len(lps) if lps else float("-inf"), time.perf_counter() - t)
