"""Train an adapter that works with a short encoder window (whisper.cpp ``audio_ctx``), so the CPU encoder only runs
over the audio it has instead of a padded 30 s (A3 in the sonic task list; ACFT-like, but plain cross-entropy).

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. .venv/bin/python -m sonic.train_short --manifest data/processed/trim/manifest.csv \
        --dora --rank 32 --batch-size 4 --grad-accum 2 --grad-checkpointing --patience 10 --out models/lora-dora-r32-short

Same trainer as ``ivg-train`` (all its options); two patches: every batch's mel is cropped to a window drawn from
``WINDOWS`` ("fit" = the longest clip in the batch plus 1 s), and the encoder accepts that shorter input with the
matching slice of its positional embeddings, which is what whisper.cpp does with ``audio_ctx``. Mixing windows,
including the full 1500, keeps the adapter usable at any window. Dev scoring goes through the same crop.
"""

from __future__ import annotations

import math
import random

import torch
from transformers.models.whisper import modeling_whisper as mw

from irregular_voice_google import train

WINDOWS = ["fit", 512, 768, 1024, 1500]
QUANTUM, MARGIN_S, FLOOR = 64, 1.0, 128
_rng = random.Random(20261009)


class SlicedPositions(torch.nn.Module):
    """Stands in for ``embed_positions`` during one forward: the first ``n`` sinusoid rows."""

    def __init__(self, emb: torch.nn.Embedding, n: int):
        super().__init__()
        self.emb, self.num_embeddings = emb, n

    def forward(self, positions):
        return self.emb.weight[: self.num_embeddings]


_encoder_forward = mw.WhisperEncoder.forward


def short_forward(self, input_features, *args, **kwargs):
    n = input_features.shape[-1] // (self.conv1.stride[0] * self.conv2.stride[0])
    if n == self.embed_positions.num_embeddings:
        return _encoder_forward(self, input_features, *args, **kwargs)
    full_n, emb = self.config.max_source_positions, self.embed_positions
    self.config.max_source_positions = n
    object.__setattr__(self, "embed_positions", SlicedPositions(emb, n))   # not registered: no state-dict change
    try:
        return _encoder_forward(self, input_features, *args, **kwargs)
    finally:
        del self.__dict__["embed_positions"]                               # back to the registered module
        self.config.max_source_positions = full_n


def window_for(max_samples: int) -> int:
    choice = _rng.choice(WINDOWS)
    if choice == "fit":
        need = math.ceil((max_samples / 16000 + MARGIN_S) * 50)
        choice = max(FLOOR, -(-need // QUANTUM) * QUANTUM)
    return min(int(choice), 1500)


_batcher_call = train.Batcher.__call__


def cropped_call(self, utterances, rng=None, p: float = 0.0):
    batch = _batcher_call(self, utterances, rng, p)
    longest = max(len(self.audio[u.audio]) for u in utterances)
    n = window_for(longest)
    batch["input_features"] = batch["input_features"][..., : 2 * n]
    return batch


def main() -> None:
    mw.WhisperEncoder.forward = short_forward
    train.Batcher.__call__ = cropped_call
    train.main()


if __name__ == "__main__":
    main()
