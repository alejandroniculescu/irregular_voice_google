"""Turns B-Czarnetzki/dysarthric_german (Hugging Face) into wav files plus a manifest.

Other German speakers with dysarthria, to check whether an adapter carries
over beyond the one speaker it was trained on. Everything goes under the
git-ignored data/external/. The dataset card names no source or license:
use it for internal comparisons only.

    uvx --from huggingface_hub hf download B-Czarnetzki/dysarthric_german \
        --include "data/*.parquet" --repo-type dataset --local-dir data/external/dysarthric_german
    uv run --with pyarrow --with soundfile python scripts/external_dysarthric_german.py
"""

import csv
import io
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path("data/external/dysarthric_german")


def main() -> None:
    (ROOT / "wav").mkdir(exist_ok=True)
    n = 0
    with open(ROOT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        out.writerow(["audio", "text", "split", "speaker"])
        for parquet in sorted((ROOT / "data").glob("*.parquet")):  # whichever files were downloaded
            split = parquet.name.split("-")[0]  # the dataset's own train/test split
            for row in pq.read_table(parquet).to_pylist():
                audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]))
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                name = Path(row["path"]).stem + ".wav"
                sf.write(ROOT / "wav" / name, audio, rate, subtype="PCM_16")
                out.writerow([f"wav/{name}", row["text"].strip(), split, "unknown"])  # the card names no speakers
                n += 1
    print(f"{n} clips -> {ROOT / 'manifest.csv'}")


if __name__ == "__main__":
    main()
