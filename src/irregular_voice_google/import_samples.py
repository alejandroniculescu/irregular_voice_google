"""Import a ``voice_samples_*`` export into ``data/`` and write our manifest.

Reads the export's phrase manifest (``data/manifests/adaptation.tsv``) and the
reviewed command/number segments (``reports/ha_phrase_segments/*/review_corrected.tsv``),
copies the audio under ``data/raw/<export name>/`` and writes one manifest CSV.
Existing splits are kept (``test_adapt`` -> ``test``); rows without one get the
transcript-hash split.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

SPLIT_MAP = {"train": "train", "dev": "dev", "test": "test", "test_adapt": "test"}


def _read_tsv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def collect(export: Path) -> list[dict]:
    """Rows of (source audio, text, split, speaker, source tag) from an export folder."""
    rows = []
    for r in _read_tsv(export / "data/manifests/adaptation.tsv"):
        rows.append({"src": export / r["audio_path"], "text": r["transcript"].strip(),
                     "split": SPLIT_MAP[r["split"]], "speaker": r["speaker_id"], "source": "phrases"})
    speaker = rows[0]["speaker"] if rows else "patient"
    for review in sorted(export.glob("reports/ha_phrase_segments/*/review_corrected.tsv")):
        for r in _read_tsv(review):
            rows.append({"src": export / r["numbered_audio_path"], "text": r["transcription"].strip(),
                         "split": "", "speaker": speaker, "source": f"segments:{review.parent.name}"})
    return rows


def import_export(export: str | Path, data_dir: str | Path = "data", manifest: str | Path = "data/manifest.csv") -> int:
    export, data_dir, manifest = Path(export), Path(data_dir), Path(manifest)
    dest_root = data_dir / "raw" / export.name
    out_rows = []
    for row in collect(export):
        dest = dest_root / row["src"].relative_to(export)
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row["src"], dest)
        rel = dest.resolve().relative_to(manifest.parent.resolve(), walk_up=True)
        out_rows.append({"audio": rel.as_posix(), "text": row["text"], "split": row["split"],
                         "speaker": row["speaker"], "source": row["source"]})
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio", "text", "split", "speaker", "source"])
        writer.writeheader()
        writer.writerows(out_rows)
    return len(out_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("export", help="Path to a voice_samples_* folder.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="data/manifest.csv")
    args = parser.parse_args()
    n = import_export(args.export, args.data_dir, args.out)
    print(f"wrote {n} rows to {args.out}")
