from irregular_voice_google.text import normalize


def test_normalize_keeps_umlauts_and_strips_punctuation():
    assert normalize("Nach München, bitte!") == "nach münchen bitte"
    assert normalize("Straße") == "straße"


def test_normalize_splits_hyphens():
    assert normalize("Hin- und Rückflug") == "hin und rückflug"


def test_normalize_unifies_unicode_forms():
    decomposed = "München"
    assert normalize(decomposed) == normalize("München")


def test_normalize_spells_out_numbers():
    assert normalize("Am 12. Oktober") == "am zwölften oktober"
    assert normalize("Am 1. März um 14:30") == "am ersten märz um vierzehn dreißig"
    assert normalize("2 Personen") == normalize("zwei Personen")
    assert normalize("Ich brauche 3.") == "ich brauche drei"
    assert normalize("21") == "einundzwanzig"
    assert normalize("2026") == "zweitausendsechsundzwanzig"
    assert normalize("101") == "hunderteins"
