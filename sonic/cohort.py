"""Cohort uniqueness points of a closed command set (C1 in docs/open_questions.md): after how many phones is a
command the only one left, and how many milliseconds before its end is that at his speaking rate.

    PYTHONPATH=src:. .venv-sonic/bin/python -m sonic.cohort

Prints aggregates only; the command lists are public resources, his recordings contribute only a rate.
"""

from __future__ import annotations

import csv
import itertools
import json
import wave
from pathlib import Path

import numpy as np

from irregular_voice_google.home import ACTIONS, IN
from irregular_voice_google.per import VOWELS, gruut_g2p, phone_class

ROOT = Path(__file__).resolve().parent.parent


ORDERS = ("device room action", "room device action", "device action room", "action device room")


def home_commands(order: str = ORDERS[0]) -> list[str]:
    """Every device x room x action, slots spoken in ``order`` (today's phrasing: "Licht im Flur an")."""
    rooms = json.loads((ROOT / "resources/home_de.json").read_text(encoding="utf-8"))["room"]["values"]
    out = []
    for d, acts in ACTIONS.items():
        for r in rooms:
            for a in acts:
                slots = {"device": d, "room": IN.get(r, f"im {r}"), "action": a}
                out.append(" ".join(slots[k] for k in order.split()))
    return out


def short_commands() -> list[str]:
    lines = (ROOT / "resources/commands_de.txt").read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]


def uniqueness_points(seqs: list[list[str]]) -> list[int]:
    """Per sequence: the number of phones after which no other sequence shares its prefix (its length if never)."""
    out = []
    for i, s in enumerate(seqs):
        k = 0
        while k < len(s) and any(j != i and t[:k + 1] == s[:k + 1] for j, t in enumerate(seqs)):
            k += 1
        out.append(min(k + 1, len(s)))
    return out


def vowels_as_wildcards(seq: list[str]) -> list[str]:
    return ["V" if phone_class(p)[0] == "vowel" else p for p in seq]


def speaking_rate(manifest: Path, g2p, dbfs: float = -40.0) -> float:
    """Reference phones per second of voiced span (energy onset to offset), pooled over the test split."""
    phones = seconds = 0.0
    with manifest.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] != "test":
                continue
            with wave.open(str(manifest.parent / r["audio"])) as w:
                a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
            fr = a[: len(a) // 160 * 160].reshape(-1, 160)
            loud = np.nonzero(np.sqrt((fr ** 2).mean(axis=1)) > 10 ** (dbfs / 20))[0]
            if loud.size:
                seconds += (loud[-1] - loud[0] + 1) * 0.01
                phones += len(g2p(r["text"]))
    return phones / seconds


def main() -> None:
    g2p = gruut_g2p()
    rate = speaking_rate(ROOT / "data/processed/trim/manifest.csv", g2p)
    print(f"his speaking rate: {rate:.1f} phones/s ({1000 / rate:.0f} ms per phone)")
    res = {"rate_phones_per_s": rate}
    sets = [(f"home: {o}", home_commands(o)) for o in ORDERS] + [("short", short_commands())]
    for name, cmds in sets:
        seqs = [g2p(c) for c in cmds]
        for variant, ss in (("exact", seqs), ("vowel-wildcard", [vowels_as_wildcards(s) for s in seqs])):
            up = uniqueness_points(ss)
            left = np.array([len(s) - u for s, u in zip(ss, up)])
            ms = left / rate * 1000
            never = sum(u == len(s) and any(j != i and t == s for j, t in enumerate(ss)) for i, (s, u) in enumerate(zip(ss, up)))
            res[f"{name}/{variant}"] = {"n": len(cmds), "median_ms": float(np.median(ms)), "p25_ms": float(np.percentile(ms, 25)),
                                        "p75_ms": float(np.percentile(ms, 75)), "share_ge_500ms": float(np.mean(ms >= 500)),
                                        "indistinguishable": int(never), "mean_len": float(np.mean([len(s) for s in ss]))}
            r = res[f"{name}/{variant}"]
            print(f"{name:<24} {variant:<15} n={r['n']:3d}  saving median {r['median_ms']:4.0f} ms  [p25 {r['p25_ms']:4.0f}, p75 {r['p75_ms']:4.0f}]"
                  f"  ≥500 ms on {r['share_ge_500ms']:.0%}  indistinguishable {r['indistinguishable']}  (mean {r['mean_len']:.1f} phones)")
    out = ROOT / "results/sonic/cohort.json"; out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
