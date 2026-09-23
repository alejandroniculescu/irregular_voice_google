import struct

import pytest

from irregular_voice_google.manifest import build_manifest, load_manifest
from irregular_voice_google.recorder import Recorder, load_prompts


def wav_bytes(n_samples=1600, rate=16000):
    data = b"\x00\x00" * n_samples
    header = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(data), b"WAVE", b"fmt ", 16, 1, 1,
                         rate, rate * 2, 2, 16, b"data", len(data))
    return header + data


@pytest.fixture
def recorder(tmp_path):
    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text("# comment\n\nNach Wien.\nJa.\n", encoding="utf-8")
    return Recorder(load_prompts(prompts_file), tmp_path / "raw", "patient", target=2)


def test_prompt_ids_are_stable_and_skip_comments(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("# x\nJa.\n", encoding="utf-8")
    [p] = load_prompts(f)
    assert p["text"] == "Ja." and p["id"] == load_prompts(f)[0]["id"]


def test_save_writes_wav_and_transcript_and_reorders_queue(recorder):
    first = recorder.state()["prompts"][0]
    assert first["text"] == "Nach Wien."

    wav = recorder.save(first["id"], wav_bytes())
    assert wav.read_bytes()[:4] == b"RIFF"
    assert wav.with_suffix(".txt").read_text(encoding="utf-8").strip() == "Nach Wien."

    state = recorder.state()
    assert [p["text"] for p in state["prompts"]] == ["Ja.", "Nach Wien."]
    assert state["prompts"][1]["count"] == 1


def test_rejects_unknown_prompt_and_non_wav(recorder):
    pid = recorder.state()["prompts"][0]["id"]
    with pytest.raises(KeyError):
        recorder.save("deadbeef", wav_bytes())
    with pytest.raises(ValueError):
        recorder.save(pid, b"not audio")


def test_recordings_feed_the_manifest(recorder, tmp_path):
    for p in recorder.state()["prompts"]:
        recorder.save(p["id"], wav_bytes())
    out = tmp_path / "data" / "manifest.csv"
    assert build_manifest(tmp_path / "raw", out) == 2
    assert {u.text for u in load_manifest(out)} == {"Nach Wien.", "Ja."}
