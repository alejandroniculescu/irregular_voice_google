"""Kölner Phonetik: a German sound code, so near-misses still find their list value.

The adapter's typical slips keep the consonant skeleton ("Sieben" -> "Seben",
both ``816``), so comparing codes instead of spellings recovers them. Vowels
are dropped (except a leading one), letters that sound alike share a digit and
repeated digits collapse.
"""

from __future__ import annotations

import re

_SIMPLE = {**dict.fromkeys("aeijouy", "0"), "b": "1", **dict.fromkeys("dt", "2"), **dict.fromkeys("fvw", "3"),
           **dict.fromkeys("gkq", "4"), "l": "5", **dict.fromkeys("mn", "6"), "r": "7", **dict.fromkeys("sz", "8")}
_FOLD = str.maketrans({"ä": "a", "ö": "o", "ü": "u", "ß": "s"})


def _digit(w: str, i: int) -> str:
    c, prev, nxt = w[i], w[i - 1] if i else "", w[i + 1] if i + 1 < len(w) else ""
    if c == "h":
        return ""
    if c == "p":
        return "3" if nxt == "h" else "1"
    if c in "dt":
        return "8" if nxt in ("c", "s", "z") and nxt else "2"
    if c == "x":
        return "8" if prev in ("c", "k", "q") and prev else "48"
    if c == "c":
        if i == 0:
            return "4" if nxt in "ahkloqrux" and nxt else "8"
        return "4" if nxt in "ahkoqux" and nxt and prev not in ("s", "z") else "8"
    return _SIMPLE.get(c, "")


def code(text: str) -> str:
    """Kölner Phonetik code of a word or phrase (words are coded separately, joined by spaces)."""
    out = []
    for word in re.findall(r"[a-z]+", text.lower().translate(_FOLD)):
        digits = "".join(_digit(word, i) for i in range(len(word)))
        collapsed = re.sub(r"(\d)\1+", r"\1", digits)
        out.append(collapsed[:1] + collapsed[1:].replace("0", ""))
    return " ".join(w for w in out if w)
