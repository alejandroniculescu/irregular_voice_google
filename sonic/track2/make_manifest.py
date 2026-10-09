"""Write the SAPC2 kit's manifests (``id,audio_filepath[,mfa_speech_start]``) for our test split, so the kit's
``local_decode.py``, ``compute_latency.py`` and ``track2_reject_check.py`` run on Christian's clips.

    python sonic/track2/make_manifest.py --out results/sonic/kit

``mfa_speech_start`` is normally a forced-alignment onset; here it is an energy onset (first 10 ms frame above
-40 dBFS), an approximation: reject rule 1 measured against it is indicative, not official. Also writes a
``reference.csv`` (id,reference) for scoring, which holds transcripts and so stays under git-ignored ``results/``.
"""

from __future__ import annotations

import argparse
import csv
import wave
from pathlib import Path

import numpy as np


def onset(path: Path, dbfs: float = -40.0) -> float:
    with wave.open(str(path)) as f:
        a = np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(np.float32) / 32768
    frames = a[: len(a) // 160 * 160].reshape(-1, 160)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    loud = np.nonzero(rms > 10 ** (dbfs / 20))[0]
    return float(loud[0] * 0.01) if loud.size else 0.0


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default="data/processed/trim/manifest.csv")
    parser.add_argument("--out", default="results/sonic/kit")
    args = parser.parse_args(argv)
    src = Path(args.manifest); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with src.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    root = src.parent.resolve()
    with (out / "Test.csv").open("w", newline="") as a, (out / "Test_streaming.csv").open("w", newline="") as b, \
            (out / "reference.csv").open("w", newline="", encoding="utf-8") as c:
        wa, wb, wc = csv.writer(a), csv.writer(b), csv.writer(c)
        wa.writerow(["id", "audio_filepath"]); wb.writerow(["id", "audio_filepath", "mfa_speech_start"]); wc.writerow(["id", "reference"])
        for i, r in enumerate(rows):
            uid = f"test{i:03d}"
            wa.writerow([uid, r["audio"]]); wb.writerow([uid, r["audio"], f"{onset(root / r['audio']):.2f}"]); wc.writerow([uid, r["text"]])
    print(f"{len(rows)} clips -> {out} (data root {root})")


if __name__ == "__main__":
    main()
