"""Reference-free routers over several systems' hypotheses for the same clips (R1 in docs/open_questions.md).

    PYTHONPATH=src:. python -m sonic.route results/20261009-132109 --pivot "models__lora-dora-r32+snap" \
        --others "models__lora-r16+snap" "models__lora-aug-synth+snap" "models__lora-aug+snap"

Each router sees only hypotheses, never the reference: medoid (the hypothesis closest in phones to the others),
rover (word-level vote aligned to the pivot, ties to the pivot), switch (the pivot unless two other systems agree
word for word against it). Scored by PER against the pivot with the paired bootstrap, and as the share of the
oracle gap closed. Writes the routed CSVs next to the inputs as ``route__<name>.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import jiwer

from irregular_voice_google.per import counts, gruut_g2p, load_rows, oracle, paired_bootstrap, per, per_rows, profile
from irregular_voice_google.text import normalize


def medoid(hyps: list[str], g2p, pivot: int = 0) -> int:
    """Index of the hypothesis with the smallest total phone edit distance to the others; ties to the pivot."""
    ph = [g2p(h) for h in hyps]
    cost = [sum(counts(ph[i], ph[j])[1] for j in range(len(hyps)) if j != i) for i in range(len(hyps))]
    best = min(cost)
    return pivot if cost[pivot] == best else cost.index(best)


def rover(hyps: list[str], pivot: int = 0) -> str:
    """Word-level vote on the pivot's slots: each other hypothesis contributes the word aligned to each pivot word
    (or nothing, on a deletion); its insertions are dropped. Majority wins, ties keep the pivot's word."""
    pw = normalize(hyps[pivot]).split()
    if not pw:
        return hyps[pivot]
    slots = [[w] for w in pw]
    for k, h in enumerate(hyps):
        if k == pivot:
            continue
        hw = normalize(h).split()
        if not hw:
            for s in slots:
                s.append("")
            continue
        for c in jiwer.process_words(" ".join(pw), " ".join(hw)).alignments[0]:
            for i in range(c.ref_start_idx, c.ref_end_idx):
                if c.type == "equal":
                    slots[i].append(pw[i])
                elif c.type == "substitute":
                    slots[i].append(hw[c.hyp_start_idx + (i - c.ref_start_idx)])
                elif c.type == "delete":
                    slots[i].append("")
    out = []
    for s in slots:
        votes = Counter(s).most_common()
        top = votes[0][1]
        word = s[0] if Counter(s)[s[0]] == top else votes[0][0]
        if word:
            out.append(word)
    return " ".join(out)


def switch(hyps: list[str], pivot: int = 0) -> int:
    """The pivot unless at least two other hypotheses are identical to each other and differ from it."""
    norm = [normalize(h) for h in hyps]
    others = Counter(n for k, n in enumerate(norm) if k != pivot)
    text, n = others.most_common(1)[0] if others else ("", 0)
    if n >= 2 and text != norm[pivot]:
        return next(k for k, t in enumerate(norm) if k != pivot and t == text)
    return pivot


def scores(rows: list[dict], g2p) -> dict:
    """Corpus WER, CER, PER and the vowel error rate (sub+del on reference vowels, the class the adapter learned
    least) for one system's rows."""
    ref = [normalize(r["reference"]) for r in rows]
    hyp = [normalize(r["hypothesis"]) for r in rows]
    v = profile(rows, g2p).get("vowel", {"rate": None})
    return {"wer": jiwer.wer(ref, hyp), "cer": jiwer.cer(ref, hyp), "per": per(per_rows(rows, g2p)), "vowel": v["rate"]}


def fmt(s: dict) -> str:
    return f"WER {s['wer']:6.1%}  CER {s['cer']:5.1%}  PER {s['per']:5.1%}  vowel {s['vowel']:5.1%}"


def cascade(decode, n_experts: int, threshold: float) -> tuple[str, list[int]]:
    """Confidence routing with early exit: ``decode(k)`` returns (text, mean log-prob) of expert k. Expert 0 decodes
    first; if its log-prob is at least ``threshold`` it is the answer, else every expert decodes and the most
    confident wins. Returns the text and the experts that ran."""
    text, lp = decode(0)
    if lp >= threshold or n_experts == 1:
        return text, [0]
    best = (lp, 0, text)
    for k in range(1, n_experts):
        t, l = decode(k)
        if l > best[0]:
            best = (l, k, t)
    return best[2], list(range(n_experts))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_dir")
    parser.add_argument("--pivot", required=True, help="CSV stem of the best single system")
    parser.add_argument("--others", nargs="+", required=True, help="CSV stems of the other systems")
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args(argv)
    d = Path(args.results_dir)
    stems = [args.pivot, *args.others]
    tables = [load_rows(d / f"{s}.csv") for s in stems]
    assert all([r["audio"] for r in t] == [r["audio"] for r in tables[0]] for t in tables), "clip order differs"
    g2p = gruut_g2p()

    routed = {"medoid": [], "rover": [], "switch": []}
    for i, base in enumerate(tables[0]):
        hyps = [t[i]["hypothesis"] for t in tables]
        routed["medoid"].append({**base, "hypothesis": hyps[medoid(hyps, g2p)]})
        routed["rover"].append({**base, "hypothesis": rover(hyps)})
        routed["switch"].append({**base, "hypothesis": hyps[switch(hyps)]})

    arrays = {s: per_rows(t, g2p) for s, t in zip(stems, tables)}
    o = oracle(arrays)
    A = arrays[args.pivot]
    gap = per(A) - o["oracle_per"]
    res = {"pivot": args.pivot, "others": args.others, "oracle_per": o["oracle_per"], "pivot_per": per(A), "routers": {}}
    for s, t in zip(stems, tables):
        res.setdefault("systems", {})[s] = scores(t, g2p)
        print(f"{s:<32} {fmt(res['systems'][s])}")
    print(f"oracle PER {o['oracle_per']:.1%}  (gap from pivot {gap:.1%})")
    for name, rows in routed.items():
        B = per_rows(rows, g2p)
        q = paired_bootstrap(A, B, args.draws)
        changed = sum(normalize(r["hypothesis"]) != normalize(b["hypothesis"]) for r, b in zip(rows, tables[0]))
        closed = (per(A) - per(B)) / gap if gap else float("nan")
        res["routers"][name] = {**q, **scores(rows, g2p), "gap_closed": closed, "utterances_changed": changed}
        print(f"route {name:<26} {fmt(res['routers'][name])}")
        print(f"  PER diff {q['diff']:+.3f} [{q['ci95'][0]:+.3f}, {q['ci95'][1]:+.3f}]  "
              f"P(diff>=0) {q['p_diff_ge_0']:.3f}  gap closed {closed:+.0%}  changed {changed}")
        with (d / f"route__{name}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    (d / "route.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {d / 'route.json'}")


if __name__ == "__main__":
    main()
