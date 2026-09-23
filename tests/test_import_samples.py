from irregular_voice_google.import_samples import import_export
from irregular_voice_google.manifest import load_manifest


def _wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF\0\0\0\0WAVE")


def test_import_export_keeps_splits_and_copies_audio(tmp_path):
    export = tmp_path / "voice_samples_x"
    _wav(export / "data/adaptation_audio/a.wav")
    _wav(export / "data/adaptation_audio/b.wav")
    _wav(export / "artifacts/seg/001.wav")
    (export / "data/manifests").mkdir(parents=True)
    (export / "data/manifests/adaptation.tsv").write_text(
        "utt_id\taudio_path\ttranscript\tsplit\tspeaker_id\n"
        "a\tdata/adaptation_audio/a.wav\tHallo Welt.\ttrain\tspeaker_1\n"
        "b\tdata/adaptation_audio/b.wav\tGuten Tag.\ttest_adapt\tspeaker_1\n", encoding="utf-8")
    review = export / "reports/ha_phrase_segments/rec/review_corrected.tsv"
    review.parent.mkdir(parents=True)
    review.write_text("audio_num\ttranscription\tnumbered_audio_path\n1\tEins.\tartifacts/seg/001.wav\n", encoding="utf-8")

    manifest = tmp_path / "data/manifest.csv"
    assert import_export(export, tmp_path / "data", manifest) == 3
    utts = load_manifest(manifest)
    assert [u.split for u in utts[:2]] == ["train", "test"]
    assert utts[2].text == "Eins." and utts[2].speaker == "speaker_1"
    assert all(u.audio.exists() and "voice_samples_x" in u.audio.parts for u in utts)
    assert all(tmp_path / "data/raw" in u.audio.parents for u in utts)
