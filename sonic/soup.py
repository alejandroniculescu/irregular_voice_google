"""A model soup of several adapters: merge each into the base, average the full weights (optionally weighted), and
write one ggml model, so a mixture of experts costs a single encoder pass (MAS-LoRA, Interspeech 2025: uniformly
averaged experts beat each expert and a learned router).

    .venv/bin/python -m sonic.soup --adapters models/lora-dora-r32 models/lora-r16 models/lora-aug-synth \
        --out models/ggml/soup-dora-r16-synth-f16.bin            # on ahms; quantize with whisper-quantize

Averaging merged weights works across LoRA and DoRA alike, since both become plain weight matrices once merged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from irregular_voice_google.ggml import load_model, write_ggml


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapters", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, help="one per adapter (default: uniform)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    w = args.weights or [1.0] * len(args.adapters)
    assert len(w) == len(args.adapters)
    w = [x / sum(w) for x in w]

    soup, processor, avg = None, None, None
    for a, wi in zip(args.adapters, w):
        model, proc = load_model(a, dtype=torch.float32)
        sd = model.state_dict()
        if avg is None:
            soup, processor = model, proc
            avg = {k: v.clone() * wi if v.is_floating_point() else v.clone() for k, v in sd.items()}
        else:
            for k, v in sd.items():
                if v.is_floating_point():
                    avg[k] += v * wi
            del model
        print(f"added {a} (weight {wi:.3f})", flush=True)
    soup.load_state_dict(avg)
    out = Path(args.out)
    write_ggml(soup.eval(), processor, out)
    out.with_suffix(".json").write_text(json.dumps({"soup": args.adapters, "weights": w}, indent=2), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
