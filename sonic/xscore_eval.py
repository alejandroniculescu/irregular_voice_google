"""Evaluate cross-scoring (R2 in docs/open_questions.md): for each clip, the candidate with the higher sum of the
judges' z-scores wins; z uses each judge's mean and SD over its dev scores of all candidates.

    PYTHONPATH=src:. .venv-sonic/bin/python -m sonic.xscore_eval results/sonic/r2

Expects ``dev_<cand>.csv`` / ``test_<cand>.csv`` (bench outputs) and ``judge_<name>.csv`` (``sonic.xscore``) in the
directory. Prints numbers only.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from irregular_voice_google.per import gruut_g2p
from sonic.route import fmt, scores


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dir")
    parser.add_argument("--weights", nargs="*", type=float, default=[0, 0.25, 0.5, 1, 2, 4],
                        help="secondary: weight on the parakeet judge (whisper judge weight 1)")
    args = parser.parse_args(argv)
    d = Path(args.dir)
    cands = sorted({p.stem.split("_", 1)[1] for p in d.glob("dev_*.csv")})
    judges = sorted(p.stem.split("_", 1)[1] for p in d.glob("judge_*.csv"))
    rows = {(s, c): read(d / f"{s}_{c}.csv") for s in ("dev", "test") for c in cands}
    score = defaultdict(dict)                       # judge -> (audio, cand) -> score
    for j in judges:
        for r in read(d / f"judge_{j}.csv"):
            stem = r["candidate"]
            score[j][(r["audio"], stem.split("_", 1)[1], stem.split("_", 1)[0])] = float(r["score"])
    stats = {}
    for j in judges:
        v = np.array([s for (a, c, split), s in score[j].items() if split == "dev" and np.isfinite(s)])
        stats[j] = (v.mean(), v.std())
        print(f"judge {j}: dev mean {v.mean():.3f} sd {v.std():.3f}")
    g2p = gruut_g2p()

    def route(split: str, w: dict[str, float]) -> tuple[list[dict], dict[str, int]]:
        base = rows[(split, cands[0])]
        out, wins = [], defaultdict(int)
        for i, r in enumerate(base):
            def total(c):
                a = rows[(split, c)][i]["audio"]
                return sum(w[j] * (score[j][(a, c, split)] - stats[j][0]) / stats[j][1] for j in judges)
            best = max(cands, key=total)
            wins[best] += 1
            out.append({**r, "hypothesis": rows[(split, best)][i]["hypothesis"]})
        return out, dict(wins)

    for split in ("dev", "test"):
        assert all([r["audio"] for r in rows[(split, c)]] == [r["audio"] for r in rows[(split, cands[0])]] for c in cands)
        for c in cands:
            print(f"{split:4s} {c:10s} alone            {fmt(scores(rows[(split, c)], g2p))}")
        for j in judges:                            # each judge alone as the router
            out, wins = route(split, {k: float(k == j) for k in judges})
            print(f"{split:4s} judge {j:9s} only  {fmt(scores(out, g2p))}  wins {wins}")
        out, wins = route(split, {j: 1.0 for j in judges})
        print(f"{split:4s} R2 (equal weights)    {fmt(scores(out, g2p))}  wins {wins}")
    if "parakeet" in judges and "whisper" in judges:
        print("secondary: weight on the parakeet judge, chosen on dev (lowest PER, then CER, then WER), test once")
        best = None
        for wp in args.weights:
            out, wins = route("dev", {"whisper": 1.0, "parakeet": wp})
            s = scores(out, g2p)
            print(f"dev  w_parakeet={wp:<4}        {fmt(s)}  wins {wins}")
            if best is None or (s["per"], s["cer"], s["wer"]) < best[0]:
                best = ((s["per"], s["cer"], s["wer"]), wp)
        out, wins = route("test", {"whisper": 1.0, "parakeet": best[1]})
        print(f"test w_parakeet={best[1]:<4} (dev-chosen) {fmt(scores(out, g2p))}  wins {wins}")


if __name__ == "__main__":
    main()
