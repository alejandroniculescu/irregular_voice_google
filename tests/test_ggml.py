import shutil
from pathlib import Path

import pytest

from irregular_voice_google.ggml import ggml_name

JFK = Path("/opt/homebrew/opt/whisper-cpp/share/whisper-cpp/jfk.wav")


def test_tensor_names_follow_whisper_cpp():
    assert ggml_name("model.encoder.layers.0.self_attn.k_proj.weight") == "encoder.blocks.0.attn.key.weight"
    assert ggml_name("model.decoder.layers.3.encoder_attn.k_proj.weight") == "decoder.blocks.3.cross_attn.key.weight"
    assert ggml_name("model.decoder.layers.3.fc1.bias") == "decoder.blocks.3.mlp.0.bias"
    assert ggml_name("model.encoder.conv1.weight") == "encoder.conv1.weight"
    assert ggml_name("model.decoder.embed_positions.weight") == "decoder.positional_embedding"
    assert ggml_name("proj_out.weight") is None


@pytest.mark.skipif(not shutil.which("whisper-cli") or not JFK.exists(), reason="needs brew install whisper-cpp")
def test_converted_tiny_model_runs_in_whisper_cpp(tmp_path):
    pytest.importorskip("transformers")
    from irregular_voice_google import cpp
    from irregular_voice_google.ggml import load_model, write_ggml

    try:
        model, processor = load_model("openai/whisper-tiny")
    except OSError:
        pytest.skip("openai/whisper-tiny not downloaded")
    bin_ = write_ggml(model, processor, tmp_path / "tiny.bin")
    (free,) = cpp.run(bin_, [JFK], language="en")
    assert "ask not what your country can do for you" in free.text.lower()
    assert free.tokens and 0 < free.min_p <= 1
    (forced,) = cpp.run(bin_, [JFK], language="en", grammar='root ::= " " ("Hello" | "Goodbye") "."\n')
    assert forced.text in ("Hello.", "Goodbye.")
