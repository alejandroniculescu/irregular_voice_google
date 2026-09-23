import csv

from irregular_voice_google.manifest import assign_split, build_manifest, load_manifest


def test_same_sentence_always_same_split():
    assert assign_split("Nach Berlin.") == assign_split("nach berlin")


def test_split_fractions_roughly_hold():
    splits = [assign_split(f"Satz Nummer {i}") for i in range(2000)]
    assert 0.15 < splits.count("test") / 2000 < 0.25
    assert 0.05 < splits.count("dev") / 2000 < 0.15


def test_build_and_load_manifest(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "a.wav").write_bytes(b"")
    (audio_dir / "a.txt").write_text("Nach Wien.", encoding="utf-8")
    (audio_dir / "b.m4a").write_bytes(b"")  # no transcript -> skipped

    out = tmp_path / "data" / "manifest.csv"
    assert build_manifest(audio_dir, out) == 1

    [utt] = load_manifest(out)
    assert utt.audio == (audio_dir / "a.wav").resolve()
    assert utt.text == "Nach Wien."
    assert utt.split == assign_split("Nach Wien.")
    assert utt.speaker == "patient"


def test_explicit_split_is_respected(tmp_path):
    manifest = tmp_path / "m.csv"
    with manifest.open("w", newline="") as f:
        csv.writer(f).writerows([["audio", "text", "split"], ["x.wav", "Ja.", "train"]])
    assert load_manifest(manifest)[0].split == "train"


def test_train_refuses_sentences_shared_with_held_out(tmp_path):
    import pytest
    from irregular_voice_google.manifest import Utterance
    from irregular_voice_google.train import check_no_leak

    train = [Utterance(tmp_path / "a.wav", "Nach München, bitte.", "train", "p")]
    check_no_leak(train, [Utterance(tmp_path / "b.wav", "Nach Berlin.", "test", "p")])
    with pytest.raises(SystemExit):
        check_no_leak(train, [Utterance(tmp_path / "c.wav", "nach münchen bitte", "test", "p")])
