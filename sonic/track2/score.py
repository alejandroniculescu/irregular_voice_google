"""Score a ``local_decode.py`` run on our clips: WER, CER, PER, vowel error from ``Test.predict.csv`` against
``reference.csv``; TTFT, TTFT-stable, TTLT and both reject rules through the kit's own utilities. Numbers only.

    PYTHONPATH=src:. .venv-sonic/bin/python sonic/track2/score.py results/sonic/kit
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "data/ext/SAPC-template/utils")]

from compute_latency import compute_latency_from_partial_json  # noqa: E402
from track2_reject_check import compare_passes, compute_early_emission_rate  # noqa: E402

from irregular_voice_google.per import gruut_g2p  # noqa: E402
from sonic.route import fmt, scores  # noqa: E402


def main(kit: str) -> None:
    k = Path(kit)
    with (k / "reference.csv").open(encoding="utf-8") as f:
        ref = {r["id"]: r["reference"] for r in csv.DictReader(f)}
    with (k / "Test.predict.csv").open(encoding="utf-8") as f:
        rows = [{"reference": ref[r["id"]], "hypothesis": r["raw_hypos"]} for r in csv.DictReader(f)]
    acc = scores(rows, gruut_g2p())
    lat = compute_latency_from_partial_json(str(k / "Test.partial_results.json"), str(k / "Test_streaming.csv"))
    early = compute_early_emission_rate(str(k / "Test.partial_results.json"), str(k / "Test_streaming.csv"))
    passes = compare_passes(str(k / "Test.partial_results.json"), str(k / "Test.predict.csv"))
    print(f"accuracy  {fmt(acc)}")
    for key, v in lat.items():
        if isinstance(v, dict) and "median" in v and v["median"] is not None:
            print(f"{key:<20} median {v['median']:6.2f} s  p90 {v['p90']:6.2f} s  (n={v['count']})")
    print(f"rule 1 early emission: {early['n_utts_early']}/{early['n_utts_checked']} = {early['early_emission_rate']:.1%} "
          f"(reject above 5 %; onset is an energy estimate, not MFA)")
    print(f"rule 2 pass consistency: {json.dumps({x: y for x, y in passes.items() if not isinstance(y, (list, dict))})}")
    (k / "score.json").write_text(json.dumps({"accuracy": acc, "latency": lat, "rule1": early, "rule2": passes},
                                             indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
