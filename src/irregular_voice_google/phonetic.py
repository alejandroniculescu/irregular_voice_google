"""Kölner Phonetik: a German sound code, so near-misses still find their list value.

The adapter's typical slips keep the consonant skeleton ("Sieben" -> "Seben",
both ``816``), so comparing codes instead of spellings recovers them. Vowels
are dropped (except a leading one), letters that sound alike share a digit and
repeated digits collapse.

``variants`` adds looser codes for German sounds the plain code keeps apart
but that softer articulation or a dialect often merges: "Djamila" / "Jamila", "Topf" /
"Top", "Ferse" / "Fehse" (a vocalized r), "stramme" / "tramme" (a lost
initial s), "ich" / "isch", "rücksichtsloser" / "rücksichtloser" (a lost
linking s). They only widen the search for candidates; the spelling distance in
``snap.py`` still decides.
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


# Respellings for sounds that often merge in everyday or dialect German; each is applied on its own.
_LOOSE = [
    (re.compile(r"^(dsch|tsch|dj)"), "j"),  # Dschungel, Djamila ~ J-
    (re.compile(r"^pf"), "f"),  # Pfeffer ~ Feffer
    (re.compile(r"pf$"), "p"),  # Topf ~ Top (in the middle pf stays: Opfel is Apfel, not Opel)
    (re.compile(r"(?<=[aeiouy])r(?![aeiouy])"), ""),  # vocalized r: Ferse ~ Fehse
    (re.compile(r"^s(?=[ptk])"), ""),  # lost initial s: stramme ~ tramme
    (re.compile(r"(?<![s])ch"), "sch"),  # ich ~ isch
    (re.compile(r"(?<=[^aeiouy])s(?=[^aeiouyhc])"), ""),  # lost linking s: rücksichtsloser ~ rücksichtloser
]


def variants(word: str) -> set[str]:
    """The plain code of one word plus its looser codes (see ``_LOOSE``)."""
    w = word.lower().translate(_FOLD)
    out = {code(w)}
    for pattern, sub in _LOOSE:
        if (v := pattern.sub(sub, w)) != w:
            out.add(code(v))
    out.discard("")
    return out
