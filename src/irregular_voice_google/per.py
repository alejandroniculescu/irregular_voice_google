"""Phoneme error rate (PER) from an ``ivg-eval`` results folder, plus a paired utterance bootstrap between systems.

    uv run ivg-per results/20261009-132109                       # PER per system CSV
    uv run ivg-per results/20261009-132109 --pair "models__lora-dora-r32+snap" "models__lora-dora-neutts+snap"

Reference and hypothesis are normalised as in scoring, turned into German IPA phoneme strings with ``gruut``
(``de-de``; a lexicon plus rules, no espeak), and aligned phone by phone: PER = (substitutions + deletions +
insertions) / reference phones, corpus level. Words gruut cannot phonemise fall back to their letters, so an
invented word still costs its length. PER sits between WER (a word is wrong) and CER (a letter is wrong): it asks
how many *sounds* were wrong, which is the unit the adapter learns and the unit the correction step reasons in.
``gruut`` is an optional extra (``pip install -e ".[per]"``); the function takes any g2p callable for tests.
"""

from __future__ import annotations

import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path

import jiwer
import numpy as np

from irregular_voice_google.text import normalize


def gruut_g2p(lang: str = "de-de"):
    from gruut import sentences  # optional extra

    @lru_cache(maxsize=None)
    def phones_of_word(word: str) -> list[str]:
        for s in sentences(word, lang=lang):
            for w in s:
                if w.phonemes:
                    return [p for p in w.phonemes if p not in ("|", "‖")]
        return list(word)                      # unknown word: its letters, so it still costs something

    def g2p(text: str) -> list[str]:
        out = []
        for word in normalize(text).split():
            out.extend(phones_of_word(word))
        return out
    return g2p


def counts(ref: list[str], hyp: list[str]) -> tuple[int, int]:
    """(reference phones, errors) for one utterance."""
    if not ref:
        return 0, len(hyp)
    if not hyp:
        return len(ref), len(ref)
    m = jiwer.process_words(" ".join(ref), " ".join(hyp))
    return len(ref), m.substitutions + m.deletions + m.insertions


def per_rows(rows: list[dict], g2p) -> np.ndarray:
    return np.array([counts(g2p(r["reference"]), g2p(r["hypothesis"])) for r in rows], dtype=float)


def per(a: np.ndarray) -> float:
    return float(a[:, 1].sum() / a[:, 0].sum()) if a[:, 0].sum() else float("nan")


def paired_bootstrap(A: np.ndarray, B: np.ndarray, draws: int = 10000, seed: int = 20261009) -> dict:
    """PER(B) − PER(A) with a 95 % interval from resampling utterances (the same draw for both systems)."""
    rng = np.random.default_rng(seed)
    n = len(A)
    bs = np.empty(draws)
    for i in range(draws):
        k = rng.integers(0, n, n)
        bs[i] = per(B[k]) - per(A[k])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"per_a": per(A), "per_b": per(B), "diff": per(B) - per(A), "ci95": [float(lo), float(hi)],
            "p_diff_ge_0": float(np.mean(bs >= 0)), "n_utterances": n}


def attribution(stages: dict[str, np.ndarray]) -> dict:
    """Where errors are born and which expert removes them, from per-utterance (ref phones, errors) arrays of
    successive stages (e.g. raw -> snap -> llm). Per stage: phones wrong, phones fixed since the previous stage,
    phones newly broken since the previous stage (an expert can also hurt), and the corpus PER."""
    names = list(stages)
    out, prev = {}, None
    for name in names:
        a = stages[name]
        row = {"per": per(a), "errors": int(a[:, 1].sum())}
        if prev is not None:
            d = a[:, 1] - prev[:, 1]
            row["fixed"] = int(-d[d < 0].sum()); row["broken"] = int(d[d > 0].sum())
            row["utterances_improved"] = int((d < 0).sum()); row["utterances_worsened"] = int((d > 0).sum())
        out[name] = row; prev = a
    return out


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_dir")
    parser.add_argument("--pair", nargs=2, metavar=("A", "B"), help="two CSV stems to bootstrap (B minus A)")
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--out", help="JSON output (default: <results_dir>/per.json)")
    parser.add_argument("--stages", help="a system stem, e.g. models__lora-dora-r32: attribute errors over raw -> +snap -> +snap+llm")
    args = parser.parse_args(argv)
    d = Path(args.results_dir)
    g2p = gruut_g2p()
    table = {}
    for p in sorted(d.glob("*.csv")):
        rows = load_rows(p)
        a = per_rows(rows, g2p)
        table[p.stem] = {"per": per(a), "ref_phones": int(a[:, 0].sum()), "n": len(rows)}
        print(f"{p.stem:<50} PER {table[p.stem]['per']:6.1%}  ({table[p.stem]['ref_phones']} ref phones)")
    res = {"per": table}
    if args.pair:
        A, B = (per_rows(load_rows(d / f"{s}.csv"), g2p) for s in args.pair)
        res["pair"] = {"a": args.pair[0], "b": args.pair[1], **paired_bootstrap(A, B, args.draws)}
        q = res["pair"]
        print(f"{args.pair[1]} minus {args.pair[0]}: {q['diff']:+.3f} [{q['ci95'][0]:+.3f}, {q['ci95'][1]:+.3f}], "
              f"P(diff >= 0) = {q['p_diff_ge_0']:.3f}")
    if args.stages:
        files = {"raw": f"{args.stages}.csv", "snap": f"{args.stages}+snap.csv", "llm": f"{args.stages}+snap+llm.csv"}
        present = {k: d / v for k, v in files.items() if (d / v).exists()}
        res["attribution"] = attribution({k: per_rows(load_rows(p), g2p) for k, p in present.items()})
        for k, r in res["attribution"].items():
            extra = f"  fixed {r['fixed']:3d}  broken {r['broken']:3d}  (utterances {r['utterances_improved']}↑ {r['utterances_worsened']}↓)" if "fixed" in r else ""
            print(f"stage {k:<5} PER {r['per']:6.1%}  errors {r['errors']:3d}{extra}")
    out = Path(args.out) if args.out else d / "per.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
