"""Domain lexicons: one ``<slot>.txt`` per slot, one term per line, ``#`` comments.

Used to score what matters for the task — did the city, date, or airline come
through — independently of overall WER.
"""

from __future__ import annotations

import re
from pathlib import Path

from irregular_voice_google.text import normalize


def load_lexicon(directory: str | Path) -> dict[str, list[str]]:
    lexicon = {}
    for file in sorted(Path(directory).glob("*.txt")):
        terms = []
        for line in file.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                terms.append(normalize(line))
        lexicon[file.stem] = terms
    return lexicon


def _contains(text: str, term: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) is not None


def keyword_hits(reference: str, hypothesis: str, terms: list[str]) -> tuple[int, int]:
    """Return (terms in reference also found in hypothesis, terms in reference)."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    expected = [t for t in terms if _contains(ref, t)]
    return sum(_contains(hyp, t) for t in expected), len(expected)
