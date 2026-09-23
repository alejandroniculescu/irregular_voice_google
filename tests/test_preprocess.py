import subprocess
import wave

import numpy as np
import pytest

from irregular_voice_google.manifest import load_manifest
from irregular_voice_google.preprocess import load, parse_steps, process_manifest, variant_name


def _tone_with_silence(path, rate=16_000):
    t = np.arange(rate) / rate
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    x = np.concatenate([np.zeros(rate), tone, np.zeros(rate)])  # 1 s silence, 1 s tone, 1 s silence
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def test_parse_and_name_steps():
    assert parse_steps("trim, tempo=1.2,eq=6") == [("trim", None), ("tempo", 1.2), ("eq", 6.0)]
    assert variant_name("trim,tempo=1.2,eq=6") == "trim_tempo1.2_eq6"
    assert variant_name("") == "raw"
    for bad in ("loud", "tempo", "trim=3"):
        with pytest.raises(ValueError):
            parse_steps(bad)


def test_trim_and_tempo_shorten_audio(tmp_path):
    wav = tmp_path / "a.wav"
    _tone_with_silence(wav)
    assert len(load(wav)) == 48_000
    trimmed = len(load(wav, "trim"))
    assert 16_000 <= trimmed <= 16_000 + 2 * 0.2 * 16_000
    assert len(load(wav, "trim,tempo=1.25")) == pytest.approx(trimmed / 1.25, rel=0.05)


def test_process_manifest_keeps_text_and_split(tmp_path):
    _tone_with_silence(tmp_path / "a.wav")
    (tmp_path / "manifest.csv").write_text("audio,text,split,speaker\na.wav,Hallo.,test,p\n", encoding="utf-8")
    out = process_manifest(tmp_path / "manifest.csv", "trim", tmp_path / "processed")
    [u] = load_manifest(out)
    assert (u.text, u.split, u.speaker) == ("Hallo.", "test", "p")
    assert u.audio.exists() and "trim" in u.audio.parts
