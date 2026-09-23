"""Convert a HF Whisper model or ``ivg-train`` LoRA adapter to whisper.cpp's ggml format.

whisper.cpp runs inference only, so a LoRA adapter is merged into its base
model first. The file layout follows whisper.cpp's
``models/convert-h5-to-ggml.py``; the mel filter bank comes from the model's
own feature extractor (identical to OpenAI's ``mel_filters.npz``).

    uv run ivg-ggml --model primeline/whisper-large-v3-turbo-german --quantize q5_0
    uv run ivg-ggml --model models/<adapter-dir> --quantize q5_0

Converted models go to ``models/ggml/`` (git-ignored: a merged adapter is
trained on patient audio).
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

GGML_MAGIC = 0x67676D6C
_NAMES = {
    "self_attn.k_proj": "attn.key",
    "self_attn.q_proj": "attn.query",
    "self_attn.v_proj": "attn.value",
    "self_attn.out_proj": "attn.out",
    "self_attn_layer_norm": "attn_ln",
    "encoder_attn.k_proj": "cross_attn.key",
    "encoder_attn.q_proj": "cross_attn.query",
    "encoder_attn.v_proj": "cross_attn.value",
    "encoder_attn.out_proj": "cross_attn.out",
    "encoder_attn_layer_norm": "cross_attn_ln",
    "fc1": "mlp.0",
    "fc2": "mlp.2",
    "final_layer_norm": "mlp_ln",
    "encoder.layer_norm.bias": "encoder.ln_post.bias",
    "encoder.layer_norm.weight": "encoder.ln_post.weight",
    "encoder.embed_positions.weight": "encoder.positional_embedding",
    "decoder.layer_norm.bias": "decoder.ln.bias",
    "decoder.layer_norm.weight": "decoder.ln.weight",
    "decoder.embed_positions.weight": "decoder.positional_embedding",
    "decoder.embed_tokens.weight": "decoder.token_embedding.weight",
}
# Kept in f32 by whisper.cpp even in an f16 file.
_F32 = {"encoder.conv1.bias", "encoder.conv2.bias", "encoder.positional_embedding", "decoder.positional_embedding"}


def load_model(model: str, dtype: torch.dtype = torch.float32) -> tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    """HF model id or ``ivg-train`` adapter dir -> (merged model, processor)."""
    if not (Path(model) / "adapter_config.json").exists():
        return WhisperForConditionalGeneration.from_pretrained(model, dtype=dtype), WhisperProcessor.from_pretrained(model)
    from peft import PeftConfig, PeftModel

    base = WhisperForConditionalGeneration.from_pretrained(PeftConfig.from_pretrained(model).base_model_name_or_path, dtype=dtype)
    return PeftModel.from_pretrained(base, model).merge_and_unload(), WhisperProcessor.from_pretrained(model)


def ggml_name(hf_name: str) -> str | None:
    """``model.decoder.layers.3.fc1.weight`` -> ``decoder.blocks.3.mlp.0.weight``; None to skip."""
    if hf_name == "proj_out.weight":  # tied to the token embedding
        return None
    parts = hf_name.split(".")[1:]
    if parts[1] == "layers":
        module = ".".join(parts[3:-1])
        mapped = "attn.key" if module == "encoder_attn.k_proj" and parts[0] == "encoder" else _NAMES[module]
        return ".".join([parts[0], "blocks", parts[2], mapped, parts[-1]])
    name = ".".join(parts)
    return _NAMES.get(name, name)


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte <-> printable-unicode table, used by Whisper's vocab.json."""
    bs = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def write_ggml(model: WhisperForConditionalGeneration, processor: WhisperProcessor, out: str | Path) -> Path:
    """Write an f16 ggml file whisper.cpp can load (and ``whisper-quantize`` can shrink)."""
    cfg = model.config
    filters = np.asarray(processor.feature_extractor.mel_filters, dtype=np.float32).T  # (n_mels, n_fft/2+1)
    added = processor.tokenizer.get_added_vocab()  # special tokens: whisper.cpp names these itself
    vocab = sorted(((t, i) for t, i in processor.tokenizer.get_vocab().items() if t not in added), key=lambda kv: kv[1])
    if [i for _, i in vocab] != list(range(len(vocab))):
        raise ValueError("expected contiguous BPE token ids starting at 0")
    byte_decoder = {c: b for b, c in _bytes_to_unicode().items()}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        f.write(struct.pack("i" * 12, GGML_MAGIC, cfg.vocab_size, cfg.max_source_positions, cfg.d_model,
                            cfg.encoder_attention_heads, cfg.encoder_layers, cfg.max_target_positions, cfg.d_model,
                            cfg.decoder_attention_heads, cfg.decoder_layers, cfg.num_mel_bins, 1))
        f.write(struct.pack("ii", *filters.shape))
        filters.tofile(f)
        f.write(struct.pack("i", len(vocab)))
        for token, _ in vocab:
            raw = bytes(byte_decoder[c] for c in token)
            f.write(struct.pack("i", len(raw)))
            f.write(raw)
        for hf_name, tensor in model.state_dict().items():
            name = ggml_name(hf_name)
            if name is None:
                continue
            data = tensor.detach().float().squeeze().numpy()
            if name in ("encoder.conv1.bias", "encoder.conv2.bias"):
                data = data.reshape(-1, 1)
            f32 = data.ndim < 2 or name in _F32
            data = data.astype(np.float32 if f32 else np.float16)
            encoded = name.encode()
            f.write(struct.pack("iii", data.ndim, len(encoded), 0 if f32 else 1))
            f.write(struct.pack("i" * data.ndim, *reversed(data.shape)))
            f.write(encoded)
            data.tofile(f)
    return out


def quantize(src: Path, qtype: str) -> Path:
    dst = src.with_name(f"{src.stem.removesuffix('-f16')}-{qtype}.bin")
    subprocess.run(["whisper-quantize", str(src), str(dst), qtype], check=True, capture_output=True)
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="HF model id or ivg-train adapter dir")
    parser.add_argument("--out", help="output .bin (default models/ggml/<name>-f16.bin)")
    parser.add_argument("--quantize", help="also write a quantized copy, e.g. q5_0 or q8_0 (needs whisper-quantize)")
    parser.add_argument("--keep-f16", action="store_true", help="keep the f16 file after quantizing")
    args = parser.parse_args()

    out = Path(args.out or f"models/ggml/{args.model.strip('/').replace('/', '__')}-f16.bin")
    model, processor = load_model(args.model)
    write_ggml(model.eval(), processor, out)
    (out.with_suffix(".json")).write_text(json.dumps({"source": args.model}, indent=2), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB)")
    if args.quantize:
        q = quantize(out, args.quantize)
        q.with_suffix(".json").write_text(json.dumps({"source": args.model, "quantize": args.quantize}, indent=2),
                                          encoding="utf-8")
        print(f"wrote {q} ({q.stat().st_size / 1e9:.2f} GB)")
        if not args.keep_f16:
            out.unlink()
            out.with_suffix(".json").unlink()
