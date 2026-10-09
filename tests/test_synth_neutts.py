"""synth_neutts: reference selection from the train split, leakage exclusion, manifest shape, resampling."""
import csv
import math
import wave
from pathlib import Path

import numpy as np
import pytest

from irregular_voice_google import synth_neutts as sn
from irregular_voice_google.manifest import Utterance
from irregular_voice_google.preprocess import SAMPLE_RATE


def make_wav(path: Path, seconds: float, sr: int = SAMPLE_RATE) -> None:
    sn.write_wav(path, 0.1 * np.sin(2 * math.pi * 220 * np.arange(int(seconds * sr)) / sr), sr)


def test_pick_reference_prefers_longest_train_clip_in_range(tmp_path):
    for name, s in (("a.wav", 2.0), ("b.wav", 6.0), ("c.wav", 9.5), ("d.wav", 20.0), ("t.wav", 9.5)):
        make_wav(tmp_path / name, s)
    utts = [Utterance(Path("a.wav"), "kurz", "train", "s"), Utterance(Path("b.wav"), "mittel", "train", "s"),
            Utterance(Path("c.wav"), "lang", "train", "s"), Utterance(Path("d.wav"), "zu lang", "train", "s"),
            Utterance(Path("t.wav"), "test satz", "test", "s")]
    import random
    ref = sn.pick_reference(utts, tmp_path, random.Random(0))
    assert ref.text == "lang"                      # 9.5 s train clip wins; the equally long test clip is never chosen
    with pytest.raises(SystemExit):
        sn.pick_reference([utts[0], utts[3], utts[4]], tmp_path, random.Random(0))


def test_resample_and_write_wav_roundtrip(tmp_path):
    x = np.sin(2 * math.pi * 440 * np.arange(24000) / 24000).astype(np.float32)
    y = sn.resample(x, 24000, 16000)
    assert len(y) == 16000 and abs(np.sqrt(np.mean(y ** 2)) - np.sqrt(np.mean(x ** 2))) < 0.02
    sn.write_wav(tmp_path / "o.wav", y)
    with wave.open(str(tmp_path / "o.wav")) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()) == (16000, 1, 2, 16000)


def test_build_writes_manifest_in_train_format_and_skips_existing(tmp_path):
    calls = []
    def fake(text):
        calls.append(text)
        return 0.05 * np.ones(24000, dtype=np.float32), 24000
    ref = Utterance(Path("audio/ref.wav"), "Referenzsatz.", "train", "speaker_1")
    rows = sn.build(["Einen Hinflug bitte.", "Nur Handgepäck."], fake, tmp_path, "neutts-clone", ref)
    assert [r["split"] for r in rows] == ["train", "train"] and rows[0]["audio"] == "audio/0000.wav"
    with (tmp_path / "manifest.csv").open(encoding="utf-8") as f:
        got = list(csv.DictReader(f))
    assert [r["text"] for r in got] == ["Einen Hinflug bitte.", "Nur Handgepäck."] and got[0]["speaker"] == "neutts-clone"
    assert (tmp_path / "reference.txt").read_text(encoding="utf-8").splitlines()[1] == "Referenzsatz."
    # re-running renders nothing new
    sn.build(["Einen Hinflug bitte.", "Nur Handgepäck."], fake, tmp_path, "neutts-clone", ref)
    assert len(calls) == 2


def test_main_excludes_held_out_sentences(tmp_path, monkeypatch):
    make_wav(tmp_path / "ref.wav", 8.0)
    man = tmp_path / "manifest.csv"
    man.write_text("audio,text,split,speaker,original_sample_rate\nref.wav,Hallo Welt.,train,s,16000\n"
                   "ref.wav,Einen Hinflug bitte.,test,s,16000\n", encoding="utf-8")
    rendered = []
    class Fake:
        def __init__(self, *a, **k): pass
        def __call__(self, text):
            rendered.append(text); return np.zeros(2400, dtype=np.float32), 24000
    monkeypatch.setattr(sn, "NeuTTSRenderer", Fake)
    sn.main(["--manifest", str(man), "--exclude", str(man), "--n", "30", "--out", str(tmp_path / "out"), "--device", "cpu"])
    assert rendered and "Einen Hinflug bitte." not in rendered          # a test-split sentence is never synthesised
    assert "Hin und Rückflug bitte." in rendered
