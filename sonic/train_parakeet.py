"""Fine-tune Parakeet (TDT/RNN-T, transformers) on the speaker's train split, selecting on dev CER (P2 in
docs/open_questions.md). Full fine-tune, AdamW, bf16 autocast, one GPU.

    PYTHONPATH=src:. .venv-parakeet-gpu/bin/python -m sonic.train_parakeet --out models/parakeet-v3-christian

Saves the best-dev-CER model and processor to ``--out`` (git-ignored) and ``train_log.json`` beside it. Prints
numbers only. The test split is never read here.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import jiwer
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from irregular_voice_google.augment import augment
from irregular_voice_google.text import normalize
from sonic.bench_io import read_wav

SAMPLE_RATE = 16000


def split(manifest: Path, name: str) -> list[dict]:
    with manifest.open(encoding="utf-8") as f:
        return [{"audio": read_wav(manifest.parent / r["audio"]), "text": r["text"]}
                for r in csv.DictReader(f) if r["split"] == name]


def dev_cer(model, processor, dev: list[dict], device: str) -> float:
    model.eval()
    hyps = []
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        for i in range(0, len(dev), 8):
            batch = dev[i:i + 8]
            inputs = processor([u["audio"] for u in batch], sampling_rate=SAMPLE_RATE, return_tensors="pt").to(device)
            out = model.generate(**inputs)
            hyps += processor.batch_decode(getattr(out, "sequences", out), skip_special_tokens=True)
    model.train()
    return jiwer.cer([normalize(u["text"]) for u in dev], [normalize(h) or "-" for h in hyps])


def save(model, processor, out: Path) -> None:
    """save_pretrained, retried once: run 1 of P2 died at epoch 15 on a TypeError inside the stdlib regex compiler
    while transformers built its weight-renaming patterns (11 earlier saves had worked)."""
    for attempt in (1, 2):
        try:
            model.save_pretrained(out); processor.save_pretrained(out)
            return
        except TypeError as e:
            if attempt == 2:
                raise
            print(f"save failed ({e}); retrying", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="nvidia/parakeet-tdt-0.6b-v3")
    parser.add_argument("--manifest", default="data/processed/trim/manifest.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=50, help="linear warm-up steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", type=float, default=0.0,
                        help="per-effect probability of the Whisper recipe's augmentation (speed, low-pass, reverb, noise)")
    parser.add_argument("--encoder-only", action="store_true", help="freeze everything but the encoder")
    args = parser.parse_args(argv)

    random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda"
    manifest = Path(args.manifest)
    train, dev = split(manifest, "train"), split(manifest, "dev")
    processor = AutoProcessor.from_pretrained(args.base)
    model = AutoModel.from_pretrained(args.base).to(device).train()
    if args.encoder_only:
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("encoder.")
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in trainable) / 1e6:.0f} M", flush=True)
    rng = np.random.default_rng(args.seed)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))

    log = {"args": vars(args), "train": len(train), "dev": len(dev), "epochs": []}
    best, since = dev_cer(model, processor, dev, device), 0
    log["dev_cer_start"] = best
    print(f"start: dev CER {best:.1%}  ({len(train)} train, {len(dev)} dev)", flush=True)
    out = Path(args.out)
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train)
        losses = []
        for i in range(0, len(train), args.batch):
            batch = train[i:i + args.batch]
            audio = [augment(u["audio"], rng, args.augment) if args.augment else u["audio"] for u in batch]
            inputs = processor(audio, text=[u["text"] for u in batch],
                               sampling_rate=SAMPLE_RATE, return_tensors="pt").to(device)
            with torch.autocast(device, dtype=torch.bfloat16):
                loss = model(**inputs).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
        cer = dev_cer(model, processor, dev, device)
        log["epochs"].append({"epoch": epoch, "loss": float(np.mean(losses)), "dev_cer": cer})
        mark = ""
        if cer < best:
            best, since, mark = cer, 0, "  *saved"
            save(model, processor, out)
        else:
            since += 1
        print(f"epoch {epoch:2d}  loss {np.mean(losses):7.3f}  dev CER {cer:.1%}{mark}", flush=True)
        if since >= args.patience:
            break
    log["best_dev_cer"] = best
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"best dev CER {best:.1%}; model in {out}" if any(out.glob("*.safetensors")) else "no epoch beat the start")


if __name__ == "__main__":
    main()
