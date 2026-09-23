from pathlib import Path

import pytest

from irregular_voice_google.evaluate import prompt_echo
from irregular_voice_google.manifest import Utterance
from irregular_voice_google.profile import Profile, check_prompt_not_in, load_profile, prompt_text

ROOT = Path(__file__).parents[1]


def test_example_profile_loads_and_builds_prompts():
    profile = load_profile(ROOT / "resources/speakers/example/profile.json")
    assert prompt_text(profile, "none") == ""
    assert prompt_text(profile, "phrases").startswith("Ich möchte einen Flug buchen.")
    terms = prompt_text(profile, "terms")
    assert "München" in terms and "Lufthansa" in terms and "Rollstuhl" not in terms
    both = prompt_text(profile, "phrases,terms")
    assert both.startswith(terms) and both.endswith("Ja, das ist richtig.")


def test_unknown_fields_and_parts_are_rejected(tmp_path):
    (tmp_path / "p.json").write_text('{"speaker": "x", "prompt": "hi"}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_profile(tmp_path / "p.json")
    with pytest.raises(ValueError):
        prompt_text(Profile("x"), "phrases,description")


def test_prompt_must_not_contain_evaluated_sentences(tmp_path):
    utts = [Utterance(tmp_path / "a.wav", "Nach München, bitte!", "dev", "x"),
            Utterance(tmp_path / "b.wav", "Wien.", "dev", "x")]
    check_prompt_not_in("Wien, Berlin. Ich möchte fliegen.", utts)  # single-word overlap is fine
    with pytest.raises(SystemExit):
        check_prompt_not_in("Ich möchte fliegen. Nach München bitte.", utts)


def test_prompt_echo_counts_prompt_words_that_were_not_said(tmp_path):
    utts = [Utterance(tmp_path / "a.wav", "nach wien", "dev", "x")]
    assert prompt_echo(utts, ["nach wien"], "Wien, Lufthansa.") == 0.0
    assert prompt_echo(utts, ["nach wien lufthansa lufthansa"], "Wien, Lufthansa.") == 0.5
