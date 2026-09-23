from irregular_voice_google.guard import guard, max_new_tokens


def test_collapses_runaway_word():
    text = "Bitter, Kripper, Kaffee, Navi, " + "Jürgen, " * 50 + "J"
    cleaned, reason = guard(text, seconds=3.0)
    assert reason == "repeat"
    assert cleaned.count("Jürgen") == 1 and cleaned.startswith("Bitter, Kripper")


def test_collapses_repeated_phrase():
    cleaned, reason = guard("Nach Berlin bitte. " + "Vielen Dank. " * 6)
    assert (cleaned, reason) == ("Nach Berlin bitte. Vielen Dank.", "repeat")


def test_leaves_normal_speech_and_short_repeats_alone():
    for text in ("Nein, nein, ich meine München.", "Null null null eins.", "Am zwölften Oktober nach Wien."):
        assert guard(text, seconds=3.0) == (text, None)


def test_flags_too_many_words_for_the_clip():
    assert guard("eins zwei drei vier fünf sechs sieben acht", seconds=0.5)[1] == "too_fast"


def test_token_cap_scales_with_duration_and_is_bounded():
    assert max_new_tokens(1.0) < max_new_tokens(5.0) <= 440
    assert max_new_tokens(600.0) == 440
