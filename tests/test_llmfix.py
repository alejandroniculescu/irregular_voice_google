import pytest

from irregular_voice_google.llmfix import fix, gate, sounds_close
from irregular_voice_google.snap import Snapper


@pytest.fixture(scope="module")
def snapper():
    return Snapper()


def test_sounds_close():
    assert sounds_close("Schweich", "Schweiß")  # 834 vs 838
    assert sounds_close("seidig", "salzig")  # 824 vs 8584
    assert not sounds_close("seidig", "bitter")


def test_gate_keeps_only_sound_alike_real_word_swaps(snapper):
    heard = "Schweich schmeckt seidig."
    assert gate(heard, "Schweiß schmeckt salzig.", snapper) == "Schweiß schmeckt salzig."
    assert gate(heard, "Schweiß schmeckt bitter.", snapper) == "Schweiß schmeckt seidig."  # not a sound-alike
    # a rewrite that adds words next to a swap is not trusted at all
    assert gate(heard, "Der Schweiß schmeckt sehr salzig.", snapper) == heard
    assert gate(heard, "Schweiß schmeckt so salzig.", snapper) == "Schweiß schmeckt seidig."  # only 1:1 swaps
    assert gate(heard, "Schweiß.", snapper) == heard  # deletions never happen


def test_fix_uses_the_proposer(snapper):
    assert fix("Schweich schmeckt seidig.", snapper, "m", lambda t, m: "Schweiß schmeckt salzig") == \
        "Schweiß schmeckt salzig."
    assert fix("...", snapper, "m", lambda t, m: "Hallo") == "..."
