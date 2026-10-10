"""CPU bench of the whisper.cpp experts on the test split: accuracy (WER, CER, PER, vowel error) and seconds per
clip for each expert at each encoder window, then the routers over the experts' decodes.

    PYTHONPATH=src:. .venv-sonic/bin/python -m sonic.bench --ctx 0 auto 512

Writes ``results/sonic/<timestamp>/`` (git-ignored): one ``<expert>@<ctx>.csv`` per run in the ``ivg-eval`` layout
(audio, reference, hypothesis, logprob, seconds), ``route__*@<ctx>.csv`` for the routers, and ``bench.json``.
Prints numbers only, never transcripts.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from irregular_voice_google.per import gruut_g2p
from sonic.bench_io import read_wav, test_clips, write_csv
from sonic.engine import Expert, audio_ctx_for
from sonic.route import fmt, medoid, scores

EXPERTS = {
    "dora": "models/ggml/models__lora-dora-r32-q5_0.bin",
    "r16": "models/ggml/models__lora-r16-q5_0.bin",
    "synth": "models/ggml/models__lora-aug-synth-q5_0.bin",
}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default="data/processed/trim/manifest.csv")
    parser.add_argument("--ctx", nargs="+", default=["0", "auto"], help="encoder windows: 0 (full 30 s), auto (fitted, floor 512), fit (fitted, floor 128), or N")
    parser.add_argument("--experts", nargs="+", default=list(EXPERTS),
                        help="names from EXPERTS, or name=path.bin for any other ggml model")
    parser.add_argument("--no-route", action="store_true", help="score the experts only (e.g. a quantization sweep)")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)

    clips = test_clips(Path(args.manifest))
    audio = [read_wav(Path(c["audio"])) for c in clips]
    out = Path("results/sonic") / datetime.now().strftime("%Y%m%d-%H%M%S"); out.mkdir(parents=True)
    g2p = gruut_g2p()
    res = {"clips": len(clips), "audio_seconds": sum(len(a) for a in audio) / 16000, "threads": args.threads, "runs": {}}

    decoded: dict[str, dict[str, list[dict]]] = {c: {} for c in args.ctx}
    paths = dict(EXPERTS)
    for spec in args.experts:
        if "=" in spec:
            name, path = spec.split("=", 1); paths[name] = path
    args.experts = [spec.split("=", 1)[0] for spec in args.experts]
    for name in args.experts:
        expert = Expert(paths[name], name, threads=args.threads)
        for c in args.ctx:
            rows = []
            for clip, a in zip(clips, audio):
                # auto: fitted with the 512 floor; fit: fitted down to 128 (for short-window adapters)
                ctx = None if c == "auto" else audio_ctx_for(len(a), floor=128) if c == "fit" else int(c)
                d = expert.decode(a, ctx)
                rows.append({**clip, "hypothesis": d.text, "logprob": round(d.logprob, 4), "seconds": round(d.seconds, 3)})
            decoded[c][name] = rows
            write_csv(out / f"{name}@{c}.csv", rows)
            s = scores(rows, g2p); t = [r["seconds"] for r in rows]
            res["runs"][f"{name}@{c}"] = {**s, "sec_mean": float(np.mean(t)), "sec_max": float(np.max(t))}
            print(f"{name:>5}@{c:<5} {fmt(s)}  sec/clip {np.mean(t):5.2f} (max {np.max(t):.2f})", flush=True)
        del expert

    for c, by in decoded.items():
        names = list(by)
        if len(names) < 2 or args.no_route:
            continue
        routed = {"medoid": [], "confidence": []}
        for i, clip in enumerate(clips):
            hyps = [by[n][i]["hypothesis"] for n in names]
            lps = [by[n][i]["logprob"] for n in names]
            routed["medoid"].append({**clip, "hypothesis": hyps[medoid(hyps, g2p)]})
            routed["confidence"].append({**clip, "hypothesis": hyps[int(np.argmax(lps))]})
        cost = sum(np.mean([r["seconds"] for r in by[n]]) for n in names)
        for r, rows in routed.items():
            write_csv(out / f"route__{r}@{c}.csv", rows)
            s = scores(rows, g2p)
            res["runs"][f"route:{r}@{c}"] = {**s, "sec_mean": float(cost), "experts": names}
            print(f"route:{r:<10}@{c:<5} {fmt(s)}  sec/clip {cost:5.2f} (sum of {len(names)} experts, sequential)")
    (out / "bench.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
