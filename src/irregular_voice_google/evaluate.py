"""Baseline evaluation: transcribe a manifest split with each model and score it.

Writes per-utterance CSVs and a summary.json under ``results/<timestamp>/``.
Results contain patient transcripts, so ``results/`` is git-ignored.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import jiwer
import torch
from transformers import pipeline

from irregular_voice_google.guard import guard, max_new_tokens
from irregular_voice_google.lexicon import keyword_hits, load_lexicon
from irregular_voice_google.manifest import SPLITS, Utterance, load_manifest
from irregular_voice_google.preprocess import SAMPLE_RATE, load
from irregular_voice_google.text import normalize

DEFAULT_MODELS = (
    "primeline/whisper-large-v3-turbo-german",
    "openai/whisper-large-v3-turbo",
)


def pick_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def load_asr(model: str, device: str, dtype: torch.dtype):
    """ASR pipeline for a HF model id, or for a LoRA adapter directory from ``ivg-train``."""
    if not (Path(model) / "adapter_config.json").exists():
        return pipeline("automatic-speech-recognition", model=model, device=device, dtype=dtype)
    from peft import PeftConfig, PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    base = WhisperForConditionalGeneration.from_pretrained(PeftConfig.from_pretrained(model).base_model_name_or_path, dtype=dtype)
    merged = PeftModel.from_pretrained(base, model).merge_and_unload()
    processor = WhisperProcessor.from_pretrained(model)
    return pipeline("automatic-speech-recognition", model=merged, tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor, device=device, dtype=dtype)


def transcribe(model: str, utterances: list[Utterance], device: str, dtype: torch.dtype) -> tuple[list[str], list[str | None]]:
    """Guarded hypotheses plus, per utterance, why the loop guard fired (or None)."""
    asr = load_asr(model, device, dtype)
    hypotheses, flags = [], []
    try:
        for i, utt in enumerate(utterances, 1):
            audio = load(utt.audio)
            seconds = len(audio) / SAMPLE_RATE
            generate_kwargs = {"language": "german", "task": "transcribe", "max_new_tokens": max_new_tokens(seconds)}
            out = asr({"raw": audio, "sampling_rate": SAMPLE_RATE}, generate_kwargs=generate_kwargs, return_timestamps=False)
            text, flag = guard(out["text"].strip(), seconds)
            hypotheses.append(text)
            flags.append(flag)
            print(f"  [{i}/{len(utterances)}] {text}" + (f"  [guard: {flag}]" if flag else ""))
    finally:
        del asr
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
    return hypotheses, flags


def score(utterances: list[Utterance], hypotheses: list[str], lexicon: dict[str, list[str]],
          flags: list[str | None] | None = None) -> tuple[dict, list[dict]]:
    refs = [normalize(u.text) for u in utterances]
    hyps = [normalize(h) for h in hypotheses]
    rows = []
    flags = flags or [None] * len(utterances)
    for utt, ref, hyp, raw, flag in zip(utterances, refs, hyps, hypotheses, flags):
        rows.append({
            "audio": str(utt.audio),
            "reference": utt.text,
            "hypothesis": raw,
            "wer": round(jiwer.wer(ref, hyp), 4) if ref else "",
            "guard": flag or "",
        })

    summary = {
        "n": len(utterances),
        "wer": round(jiwer.wer(refs, hyps), 4),
        "cer": round(jiwer.cer(refs, hyps), 4),
        "guard_flags": sum(f is not None for f in flags),
        "keyword_recall": {},
    }
    for slot, terms in lexicon.items():
        hit = total = 0
        for utt, raw in zip(utterances, hypotheses):
            h, t = keyword_hits(utt.text, raw, terms)
            hit, total = hit + h, total + t
        if total:
            summary["keyword_recall"][slot] = {"recall": round(hit / total, 4), "n": total}
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--split", default="test", choices=[*SPLITS, "all"])
    parser.add_argument("--model", action="append", help="HF model id or ivg-train adapter dir; repeatable. Defaults to the German + stock turbo models.")
    parser.add_argument("--lexicon", default="resources/lexicon")
    parser.add_argument("--out", default="results")
    parser.add_argument("--limit", type=int, help="Only the first N utterances (smoke tests).")
    args = parser.parse_args()

    utterances = load_manifest(args.manifest)
    if args.split != "all":
        utterances = [u for u in utterances if u.split == args.split]
    utterances = utterances[: args.limit]
    if not utterances:
        raise SystemExit(f"no utterances in split {args.split!r} of {args.manifest}")

    lexicon = load_lexicon(args.lexicon) if Path(args.lexicon).is_dir() else {}
    device, dtype = pick_device()
    out_dir = Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True)

    summaries = {}
    for model in args.model or DEFAULT_MODELS:
        print(f"\n== {model} on {len(utterances)} {args.split} utterances ({device})")
        start = time.perf_counter()
        hypotheses, flags = transcribe(model, utterances, device, dtype)
        summary, rows = score(utterances, hypotheses, lexicon, flags)
        summary["seconds"] = round(time.perf_counter() - start, 1)
        summaries[model] = summary

        with (out_dir / f"{model.strip('/').replace('/', '__')}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    run = {"manifest": args.manifest, "split": args.split, "device": device, "models": summaries}
    (out_dir / "summary.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'model':<45} {'WER':>6} {'CER':>6} {'loops':>5}  keyword recall")
    for model, s in summaries.items():
        kw = ", ".join(f"{k} {v['recall']:.0%}" for k, v in s["keyword_recall"].items())
        print(f"{model:<45} {s['wer']:>6.1%} {s['cer']:>6.1%} {s['guard_flags']:>5}  {kw}")
    print(f"\nresults: {out_dir}")
