from irregular_voice_google.synth import sentences
from irregular_voice_google.text import normalize


def test_sentences_are_distinct_and_cover_booking():
    texts = sentences(120)
    assert len(texts) == len({t for t in texts}) >= 120
    joined = normalize(" ".join(texts))
    assert "nach" in joined and "fliegen" in joined and "rollstuhl" in joined
