import pytest

from irregular_voice_google.snap import Snapper, edits


@pytest.fixture(scope="module")
def snapper():
    return Snapper(["Jara liebt Kekse."])


def test_edits():
    assert edits("geklabt", "geklappt") == 1  # b->p and the doubled p cost half each
    assert edits("geklabt", "geklebt") == 1
    assert edits("kinn", "kinn") == 0


def test_snaps_non_word_to_same_sounding_word(snapper):
    assert snapper.word("geklabt") == "geklappt"
    assert snapper.word("Opfel") == "Apfel"  # keeps the capital


def test_leaves_real_and_rare_words(snapper):
    for w in ["geklappt", "Bügelbrett", "pufft", "blechern", "Jara"]:
        assert snapper.word(w) == w


def test_leaves_non_word_without_close_match(snapper):
    assert snapper.word("Xqzrt") == "Xqzrt"


def test_text_keeps_punctuation(snapper):
    assert snapper.text("Bügelbrett geklabt, Blazer gebügelt.") == "Bügelbrett geklappt, Blazer gebügelt."
