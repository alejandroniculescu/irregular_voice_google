"""Turns the test split of B-Czarnetzki/dysarthric_german (Hugging Face) into wav files plus a manifest.

Other German speakers with dysarthria, to check whether an adapter carries
over beyond the one speaker it was trained on. Everything goes under the
git-ignored data/external/. The dataset card names no source or license:
use it for internal comparisons only.

    uvx --from huggingface_hub hf download B-Czarnetzki/dysarthric_german \
        data/test-00000-of-00001-b8a965f992d610f9.parquet --repo-type dataset \
        --local-dir data/external/dysarthric_german
    uv run --with pyarrow --with soundfile python scripts/external_dysarthric_german.py
"""

import csv
import io
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path("data/external/dysarthric_german")


def main() -> None:
    rows = pq.read_table(next((ROOT / "data").glob("test-*.parquet"))).to_pylist()
    (ROOT / "wav").mkdir(exist_ok=True)
    with open(ROOT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        out.writerow(["audio", "text", "split", "speaker"])
        for row in rows:
            audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            name = Path(row["path"]).stem + ".wav"
            sf.write(ROOT / "wav" / name, audio, rate, subtype="PCM_16")
            out.writerow([f"wav/{name}", row["text"].strip(), "test", "unknown"])  # the card names no speakers
    print(f"{len(rows)} clips -> {ROOT / 'manifest.csv'}")


if __name__ == "__main__":
    main()
