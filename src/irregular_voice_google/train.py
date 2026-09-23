"""Per-speaker LoRA fine-tune of a Whisper model on the manifest's train split.

Trains low-rank adapters on the encoder and decoder, scores the dev split after
every epoch and keeps the adapter with the lowest dev WER. The test split is
never touched; evaluate the result with ``ivg-eval --model <adapter dir>``.

Adapters are learned from patient audio, so they are written under the
git-ignored ``models/`` directory. Meant for a CUDA GPU; on a Mac only small
smoke runs (``--limit``, a tiny model) are practical.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import jiwer
import torch
import numpy as np
from peft import LoraConfig, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup

from irregular_voice_google.augment import augment
from irregular_voice_google.evaluate import pick_device
from irregular_voice_google.guard import guard, max_new_tokens
from irregular_voice_google.manifest import Utterance, load_manifest
from irregular_voice_google.preprocess import SAMPLE_RATE, load
from irregular_voice_google.text import normalize

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


def check_no_leak(train: list[Utterance], held_out: list[Utterance]) -> None:
    """Refuse to train on a sentence that also appears in dev/test."""
    overlap = {normalize(u.text) for u in train} & {normalize(u.text) for u in held_out}
    if overlap:
        raise SystemExit(f"{len(overlap)} transcripts are in both train and dev/test, e.g. {sorted(overlap)[:3]}")


class Batcher:
    """Turns utterances into Whisper input features and German-prefixed label ids."""

    def __init__(self, processor: WhisperProcessor, decoder_start_id: int):
        self.processor = processor
        self.decoder_start_id = decoder_start_id
        self.audio: dict[Path, object] = {}

    def __call__(self, utterances: list[Utterance], rng: np.random.Generator | None = None,
                 p: float = 0.0) -> dict[str, torch.Tensor]:
        """With ``rng``, each clip is augmented with probability ``p`` per effect (training only)."""
        for u in utterances:
            if u.audio not in self.audio:
                self.audio[u.audio] = load(u.audio)
        audio = [self.audio[u.audio] for u in utterances]
        if rng is not None:
            audio = [augment(a, rng, p) for a in audio]
        features = self.processor.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        ids = [self.processor.tokenizer(u.text).input_ids for u in utterances]
        # The model prepends the decoder start token itself when shifting labels.
        ids = [i[1:] if i and i[0] == self.decoder_start_id else i for i in ids]
        width = max(map(len, ids))
        labels = torch.full((len(ids), width), -100, dtype=torch.long)
        for row, i in enumerate(ids):
            labels[row, : len(i)] = torch.tensor(i)
        return {"input_features": features, "labels": labels}


def score(model, processor, batcher, utterances, device, autocast, batch_size) -> tuple[float, float, list[str]]:
    """WER, CER and hypotheses on unaugmented audio."""
    model.eval()
    hyps = []
    with torch.no_grad(), autocast():
        for start in range(0, len(utterances), batch_size):
            chunk = utterances[start : start + batch_size]
            features = batcher(chunk)["input_features"].to(device)
            seconds = [len(batcher.audio[u.audio]) / SAMPLE_RATE for u in chunk]
            out = model.generate(input_features=features, language="german", task="transcribe",
                                 max_new_tokens=max_new_tokens(max(seconds)))
            texts = processor.batch_decode(out, skip_special_tokens=True)
            hyps += [guard(t, s)[0] for t, s in zip(texts, seconds)]
    model.train()
    refs, norm = [normalize(u.text) for u in utterances], [normalize(h) for h in hyps]
    return jiwer.wer(refs, norm), jiwer.cer(refs, norm), hyps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--extra-manifest", action="append", default=[],
                        help="Extra training-only data (e.g. ivg-synth output); repeatable. Its splits are ignored.")
    parser.add_argument("--augment", type=float, default=0.0,
                        help="Probability of each training-time augmentation (0 = off; try 0.5).")
    parser.add_argument("--base", default="primeline/whisper-large-v3-turbo-german")
    parser.add_argument("--out", help="Adapter directory (default: models/<base>-lora-<timestamp>).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--patience", type=int, default=3, help="Stop after N epochs without dev improvement.")
    parser.add_argument("--limit", type=int, help="Only the first N train/dev utterances (smoke tests).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10, help="Print loss and ETA every N batches.")
    parser.add_argument("--grad-checkpointing", action="store_true", help="Trade compute for GPU memory.")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    utterances = load_manifest(args.manifest)
    train = [u for u in utterances if u.split == "train"]
    dev = [u for u in utterances if u.split == "dev"]
    train, dev = train[: args.limit], dev[: args.limit]
    extra = [u for m in args.extra_manifest for u in load_manifest(m)]
    check_no_leak(train + extra, [u for u in utterances if u.split != "train"])
    # A fixed sample of his own train clips, scored like dev: train WER far below dev WER means overfitting.
    train_probe = random.Random(args.seed).sample(train, min(len(train), len(dev)))
    train += extra
    if not train or not dev:
        raise SystemExit("need utterances in both the train and dev splits")

    device, _ = pick_device()
    # fp32 master weights; bf16 autocast on CUDA. MPS/CPU stay fp32 (smoke tests only).
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device == "cuda" else nullcontext

    processor = WhisperProcessor.from_pretrained(args.base)
    processor.tokenizer.set_prefix_tokens(language="german", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base, dtype=torch.float32,
        # SpecAugment: cheap regularisation for a few hundred utterances.
        apply_spec_augment=True, mask_time_prob=0.05, mask_feature_prob=0.05,
    )
    model.generation_config.forced_decoder_ids = None
    if args.grad_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, target_modules=LORA_TARGETS, bias="none",
    ))
    model.print_trainable_parameters()
    model.to(device).train()

    batcher = Batcher(processor, model.config.decoder_start_token_id)
    steps_per_epoch = -(-len(train) // (args.batch_size * args.grad_accum))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, steps_per_epoch * args.epochs // 10),
                                                steps_per_epoch * args.epochs)

    out = Path(args.out or f"models/{args.base.split('/')[-1]}-lora-{datetime.now():%Y%m%d-%H%M%S}")
    base_wer, base_cer, _ = score(model, processor, batcher, dev, device, autocast, args.batch_size)
    print(f"dev WER before training: {base_wer:.1%}, CER {base_cer:.1%} ({len(train)} train incl. {len(extra)} extra / {len(dev)} dev, "
          f"{device})", flush=True)
    aug_rng = np.random.default_rng(args.seed) if args.augment > 0 else None
    history, best, stale = [{"epoch": 0, "dev_wer": round(base_wer, 4), "dev_cer": round(base_cer, 4)}], base_wer, 0

    for epoch in range(1, args.epochs + 1):
        start, losses = time.perf_counter(), []
        order = random.sample(train, len(train))
        for step, i in enumerate(range(0, len(order), args.batch_size), 1):
            batch = {k: v.to(device) for k, v in batcher(order[i : i + args.batch_size], aug_rng, args.augment).items()}
            with autocast():
                loss = model(**batch).loss / args.grad_accum
            loss.backward()
            losses.append(loss.item() * args.grad_accum)
            if step % args.grad_accum == 0 or i + args.batch_size >= len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            n_steps = -(-len(order) // args.batch_size)
            if step % args.log_every == 0 or step == n_steps:
                elapsed = time.perf_counter() - start
                print(f"  epoch {epoch} step {step}/{n_steps}  loss {sum(losses[-args.log_every:]) / len(losses[-args.log_every:]):.3f}"
                      f"  {step / elapsed:.1f} it/s  ~{(n_steps - step) * elapsed / step:.0f}s left in epoch", flush=True)

        wer, cer, hyps = score(model, processor, batcher, dev, device, autocast, args.batch_size)
        train_wer, _, _ = score(model, processor, batcher, train_probe, device, autocast, args.batch_size)
        history.append({"epoch": epoch, "loss": round(sum(losses) / len(losses), 4), "dev_wer": round(wer, 4),
                        "dev_cer": round(cer, 4), "train_wer": round(train_wer, 4)})
        print(f"epoch {epoch}: loss {history[-1]['loss']:.3f}  dev WER {wer:.1%}  CER {cer:.1%}  "
              f"train WER {train_wer:.1%}  "
              f"({time.perf_counter() - start:.0f}s)  e.g. {dev[0].text!r} -> {hyps[0]!r}", flush=True)
        if wer < best:
            best, stale = wer, 0
            model.save_pretrained(out)
            processor.save_pretrained(out)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"no dev improvement for {args.patience} epochs, stopping")
                break

    info = {"base": args.base, "manifest": args.manifest, "n_train": len(train), "n_extra": len(extra), "n_dev": len(dev),
            "args": vars(args), "best_dev_wer": round(best, 4), "history": history}
    if best < base_wer:
        (out / "train_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"best dev WER {best:.1%} (was {base_wer:.1%}); adapter saved to {out}")
    else:
        print(f"no improvement over the base model ({base_wer:.1%}); nothing saved")
