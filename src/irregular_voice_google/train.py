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
from peft import LoraConfig, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup
from transformers.pipelines.audio_utils import ffmpeg_read

from irregular_voice_google.evaluate import pick_device
from irregular_voice_google.manifest import Utterance, load_manifest
from irregular_voice_google.text import normalize

SAMPLE_RATE = 16_000
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


def check_no_leak(train: list[Utterance], held_out: list[Utterance]) -> None:
    """Refuse to train on a sentence that also appears in dev/test."""
    overlap = {normalize(u.text) for u in train} & {normalize(u.text) for u in held_out}
    if overlap:
        raise SystemExit(f"{len(overlap)} transcripts are in both train and dev/test, e.g. {sorted(overlap)[:3]}")


def load_audio(path: Path):
    return ffmpeg_read(path.read_bytes(), SAMPLE_RATE)


class Batcher:
    """Turns utterances into Whisper input features and German-prefixed label ids."""

    def __init__(self, processor: WhisperProcessor, decoder_start_id: int):
        self.processor = processor
        self.decoder_start_id = decoder_start_id
        self.audio: dict[Path, object] = {}

    def __call__(self, utterances: list[Utterance]) -> dict[str, torch.Tensor]:
        for u in utterances:
            if u.audio not in self.audio:
                self.audio[u.audio] = load_audio(u.audio)
        features = self.processor.feature_extractor(
            [self.audio[u.audio] for u in utterances], sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features
        ids = [self.processor.tokenizer(u.text).input_ids for u in utterances]
        # The model prepends the decoder start token itself when shifting labels.
        ids = [i[1:] if i and i[0] == self.decoder_start_id else i for i in ids]
        width = max(map(len, ids))
        labels = torch.full((len(ids), width), -100, dtype=torch.long)
        for row, i in enumerate(ids):
            labels[row, : len(i)] = torch.tensor(i)
        return {"input_features": features, "labels": labels}


def dev_wer(model, processor, batcher, utterances, device, autocast, batch_size) -> tuple[float, list[str]]:
    model.eval()
    hyps = []
    with torch.no_grad(), autocast():
        for start in range(0, len(utterances), batch_size):
            chunk = utterances[start : start + batch_size]
            features = batcher(chunk)["input_features"].to(device)
            out = model.generate(input_features=features, language="german", task="transcribe")
            hyps += processor.batch_decode(out, skip_special_tokens=True)
    model.train()
    refs = [normalize(u.text) for u in utterances]
    return jiwer.wer(refs, [normalize(h) for h in hyps]), hyps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default="data/manifest.csv")
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
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    utterances = load_manifest(args.manifest)
    train = [u for u in utterances if u.split == "train"]
    dev = [u for u in utterances if u.split == "dev"]
    check_no_leak(train, [u for u in utterances if u.split != "train"])
    train, dev = train[: args.limit], dev[: args.limit]
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
    base_wer, _ = dev_wer(model, processor, batcher, dev, device, autocast, args.batch_size)
    print(f"dev WER before training: {base_wer:.1%} ({len(train)} train / {len(dev)} dev, {device})", flush=True)
    history, best, stale = [{"epoch": 0, "dev_wer": round(base_wer, 4)}], base_wer, 0

    for epoch in range(1, args.epochs + 1):
        start, losses = time.perf_counter(), []
        order = random.sample(train, len(train))
        for step, i in enumerate(range(0, len(order), args.batch_size), 1):
            batch = {k: v.to(device) for k, v in batcher(order[i : i + args.batch_size]).items()}
            with autocast():
                loss = model(**batch).loss / args.grad_accum
            loss.backward()
            losses.append(loss.item() * args.grad_accum)
            if step % args.grad_accum == 0 or i + args.batch_size >= len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        wer, hyps = dev_wer(model, processor, batcher, dev, device, autocast, args.batch_size)
        history.append({"epoch": epoch, "loss": round(sum(losses) / len(losses), 4), "dev_wer": round(wer, 4)})
        print(f"epoch {epoch}: loss {history[-1]['loss']:.3f}  dev WER {wer:.1%}  "
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

    info = {"base": args.base, "manifest": args.manifest, "n_train": len(train), "n_dev": len(dev),
            "args": vars(args), "best_dev_wer": round(best, 4), "history": history}
    if best < base_wer:
        (out / "train_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"best dev WER {best:.1%} (was {base_wer:.1%}); adapter saved to {out}")
    else:
        print(f"no improvement over the base model ({base_wer:.1%}); nothing saved")
