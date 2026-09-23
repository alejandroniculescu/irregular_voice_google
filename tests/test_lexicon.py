from irregular_voice_google.lexicon import keyword_hits, load_lexicon


def test_keyword_hits_whole_words_only():
    terms = ["wien", "münchen", "münster"]
    assert keyword_hits("Nach München.", "nach münster", terms) == (0, 1)
    assert keyword_hits("Von Wien nach München.", "von wien nach münchen", terms) == (2, 2)
    assert keyword_hits("Ja.", "ja", terms) == (0, 0)


def test_multiword_terms():
    assert keyword_hits("Hin- und Rückflug, bitte.", "hin und rückflug bitte", ["hin und rückflug"]) == (1, 1)


def test_repo_lexicon_loads():
    lexicon = load_lexicon("resources/lexicon")
    assert {"cities", "dates", "airlines", "booking"} <= lexicon.keys()
    assert "münchen" in lexicon["cities"]
