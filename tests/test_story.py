from pathlib import Path

from irregular_voice_google.lexicon import keyword_hits, load_lexicon
from irregular_voice_google.recorder import load_prompts

ROOT = Path(__file__).parents[1]


def test_story_uses_every_lexicon_term():
    story = " ".join(p["text"] for p in load_prompts(ROOT / "resources/prompts/story_de.txt"))
    missing = [
        term
        for terms in load_lexicon(ROOT / "resources/lexicon").values()
        for term in terms
        if keyword_hits(story, story, [term])[1] == 0
    ]
    assert missing == []
