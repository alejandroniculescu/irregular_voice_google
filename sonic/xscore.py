"""Cross-scoring (R2 in docs/open_questions.md): score other systems' hypotheses with a model acting as judge, as
log P(text | audio) per token on the same audio. Two judges:

- ``whisper``: a uniform soup of Whisper adapters merged in memory (as ``sonic.soup``), teacher-forced after the
  ``<|de|><|transcribe|><|notimestamps|>`` prefix; mean log-prob over the text tokens and end-of-text. GPU if any.
- ``parakeet``: a transformers Parakeet; −TDT loss (already per token).

    # main venv (peft, CUDA)
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. .venv/bin/python -m sonic.xscore --judge whisper \
        --adapters models/lora-dora-r32 models/lora-r16 models/lora-aug-synth --out x.csv a.csv b.csv
    PYTHONPATH=src:. .venv-parakeet/bin/python -m sonic.xscore --judge parakeet \
        --model models/parakeet-v3-christian-run3 --out y.csv a.csv b.csv

Each input CSV has audio and hypothesis columns (any bench output). Writes audio, candidate (input file stem),
score. Prints nothing per clip.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from sonic.bench_io import read_wav

SAMPLE_RATE = 16000


class WhisperJudge:
    def __init__(self, adapters: list[str]):
        from irregular_voice_google.ggml import load_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        avg, model = None, None
        for a in adapters:
            m, self.processor = load_model(a, dtype=torch.float32)
            sd = m.state_dict()
            if avg is None:
                model, avg = m, {k: v.clone() / len(adapters) if v.is_floating_point() else v.clone() for k, v in sd.items()}
            else:
                for k, v in sd.items():
                    if v.is_floating_point():
                        avg[k] += v / len(adapters)
                del m
        model.load_state_dict(avg)
        self.model = model.to(self.device).eval()
        self.tok = self.processor.tokenizer
        self.tok.set_prefix_tokens(language="german", task="transcribe", predict_timestamps=False)
        self.n_prefix = len(self.tok("").input_ids) - 1          # <|startoftranscript|><|de|><|transcribe|><|notimestamps|>

    def score(self, audio: np.ndarray, text: str) -> float:
        if not text.strip():
            return float("-inf")
        feats = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(self.device)
        ids = torch.tensor([self.tok(text.strip()).input_ids], device=self.device)   # as in training: no leading space
        with torch.inference_mode():
            logits = self.model(input_features=feats, decoder_input_ids=ids[:, :-1]).logits
        lp = torch.log_softmax(logits.float(), -1)[0, torch.arange(ids.shape[1] - 1), ids[0, 1:]]
        return float(lp[self.n_prefix - 1:].mean())               # the text tokens and end-of-text


class ParakeetJudge:
    def __init__(self, model: str, threads: int = 4):
        from sonic.parakeet import ParakeetExpert
        self.expert = ParakeetExpert(model, threads=threads)

    def score(self, audio: np.ndarray, text: str) -> float:
        return self.expert.logprob(audio, text.strip())


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--judge", choices=["whisper", "parakeet"], required=True)
    parser.add_argument("--adapters", nargs="+", help="whisper judge: adapters to average")
    parser.add_argument("--model", help="parakeet judge: model id or path")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", required=True)
    parser.add_argument("candidates", nargs="+", help="bench CSVs with audio and hypothesis columns")
    args = parser.parse_args(argv)

    judge = WhisperJudge(args.adapters) if args.judge == "whisper" else ParakeetJudge(args.model, args.threads)
    out, cache = [], {}
    for path in args.candidates:
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["audio"] not in cache:
                    cache[r["audio"]] = read_wav(Path(r["audio"]))
                out.append({"audio": r["audio"], "candidate": Path(path).stem,
                            "score": round(judge.score(cache[r["audio"]], r["hypothesis"]), 5)})
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["audio", "candidate", "score"]); w.writeheader(); w.writerows(out)
    print(f"{args.judge}: scored {len(out)} hypotheses -> {args.out}")


if __name__ == "__main__":
    main()
