"""Baseline evaluation: transcribe a manifest split with each model and score it.

Writes per-utterance CSVs and a summary.json under ``results/<timestamp>/``.
A ``--model`` ending in ``.bin`` is a whisper.cpp model from ``ivg-ggml`` and
runs through ``whisper-cli``; ``--question`` adds that question's prompt (and,
for whisper.cpp, its answer grammar).
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

from irregular_voice_google import cpp, questions
from irregular_voice_google.guard import guard, max_new_tokens
from irregular_voice_google.lexicon import keyword_hits, load_lexicon
from irregular_voice_google.manifest import SPLITS, Utterance, load_manifest
from irregular_voice_google.preprocess import SAMPLE_RATE, load
from irregular_voice_google.profile import check_prompt_not_in, load_profile, prompt_text
from irregular_voice_google.text import normalize

MAX_PROMPT_TOKENS = 200  # Whisper allows up to half its 448-token context for the prompt

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


def prompt_ids(asr, prompt: str, device: str) -> torch.Tensor:
    ids = asr.tokenizer.get_prompt_ids(prompt, return_tensors="pt")
    if len(ids) > MAX_PROMPT_TOKENS:  # keep <|startofprev|> and the most recent context
        ids = torch.cat([ids[:1], ids[-(MAX_PROMPT_TOKENS - 1):]])
    return ids.to(device)


def prompt_echo(utterances: list[Utterance], hypotheses: list[str], prompt: str) -> float:
    """Share of hypothesis words that come from the prompt but were not said."""
    prompt_words = set(normalize(prompt).split())
    echoed = total = 0
    for utt, hyp in zip(utterances, hypotheses):
        ref = set(normalize(utt.text).split())
        words = normalize(hyp).split()
        total += len(words)
        echoed += sum(w in prompt_words and w not in ref for w in words)
    return echoed / total if total else 0.0


def transcribe(model: str, utterances: list[Utterance], device: str, dtype: torch.dtype,
               prompt: str = "") -> tuple[list[str], list[str | None]]:
    """Guarded hypotheses plus, per utterance, why the loop guard fired (or None)."""
    asr = load_asr(model, device, dtype)
    prompt_kwargs = {"prompt_ids": prompt_ids(asr, prompt, device)} if prompt else {}
    hypotheses, flags = [], []
    try:
        for i, utt in enumerate(utterances, 1):
            audio = load(utt.audio)
            seconds = len(audio) / SAMPLE_RATE
            generate_kwargs = {"language": "german", "task": "transcribe", "max_new_tokens": max_new_tokens(seconds),
                               **prompt_kwargs}
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
          flags: list[str | None] | None = None, min_p: list[float] | None = None) -> tuple[dict, list[dict]]:
    refs = [normalize(u.text) for u in utterances]
    hyps = [normalize(h) for h in hypotheses]
    rows = []
    flags = flags or [None] * len(utterances)
    for i, (utt, ref, hyp, raw, flag) in enumerate(zip(utterances, refs, hyps, hypotheses, flags)):
        rows.append({
            "audio": str(utt.audio),
            "reference": utt.text,
            "hypothesis": raw,
            "wer": round(jiwer.wer(ref, hyp), 4) if ref else "",
            "guard": flag or "",
            **({"min_p": min_p[i]} if min_p else {}),
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
    parser.add_argument("--profile", help="Speaker profile JSON; its phrases/terms become Whisper's prompt.")
    parser.add_argument("--prompt-parts", default="phrases", help='"none", "phrases", "terms" or "phrases,terms".')
    parser.add_argument("--question", help="booking question from resources/questions_de.json (e.g. destination); "
                        "replaces --prompt-parts with that question's prompt, plus its grammar on whisper.cpp")
    parser.add_argument("--no-grammar", action="store_true", help="with --question: prompt only, no grammar")
    args = parser.parse_args()

    utterances = load_manifest(args.manifest)
    if args.split != "all":
        utterances = [u for u in utterances if u.split == args.split]
    utterances = utterances[: args.limit]
    if not utterances:
        raise SystemExit(f"no utterances in split {args.split!r} of {args.manifest}")

    profile = load_profile(args.profile) if args.profile else None
    question = questions.load_questions(lexicon=args.lexicon)[args.question] if args.question else None
    if question:
        prompt = questions.prompt(question, profile)
    else:
        prompt = prompt_text(profile, args.prompt_parts) if profile else ""
    grammar = questions.grammar(question) if question and not args.no_grammar else None
    if prompt:
        check_prompt_not_in(prompt, utterances)
        print(f"prompt ({len(prompt.split())} words): {prompt[:160]}{'…' if len(prompt) > 160 else ''}")
    lexicon = load_lexicon(args.lexicon) if Path(args.lexicon).is_dir() else {}
    device, dtype = pick_device()
    out_dir = Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True)

    summaries = {}
    for model in args.model or DEFAULT_MODELS:
        print(f"\n== {model} on {len(utterances)} {args.split} utterances ({device})")
        start = time.perf_counter()
        if model.endswith(".bin"):
            hypotheses, flags, min_p = cpp.transcribe(model, utterances, prompt, grammar)
        else:
            hypotheses, flags, min_p = *transcribe(model, utterances, device, dtype, prompt), None
        summary, rows = score(utterances, hypotheses, lexicon, flags, min_p)
        summary["prompt_echo"] = round(prompt_echo(utterances, hypotheses, prompt), 4) if prompt else 0.0
        summary["seconds"] = round(time.perf_counter() - start, 1)
        summaries[model] = summary

        with (out_dir / f"{model.strip('/').replace('/', '__')}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    run = {"manifest": args.manifest, "split": args.split, "device": device, "profile": args.profile,
           "prompt_parts": args.prompt_parts if prompt and not question else "none", "question": args.question,
           "grammar": grammar, "prompt": prompt, "models": summaries}
    (out_dir / "summary.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'model':<45} {'WER':>6} {'CER':>6} {'loops':>5} {'echo':>5}  keyword recall")
    for model, s in summaries.items():
        kw = ", ".join(f"{k} {v['recall']:.0%}" for k, v in s["keyword_recall"].items())
        print(f"{model:<45} {s['wer']:>6.1%} {s['cer']:>6.1%} {s['guard_flags']:>5} {s['prompt_echo']:>5.1%}  {kw}")
    print(f"\nresults: {out_dir}")
