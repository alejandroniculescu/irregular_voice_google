"""Transcript normalisation shared by scoring and data splitting."""

from __future__ import annotations

import re
import unicodedata

from irregular_voice_google.numbers import spell_numbers

_NON_WORD = re.compile(r"[^\w\s]")
# A letter on its own is spoken by its name ("scharfes ß" is read "scharfes Es").
_LETTERS = {"b": "be", "c": "ce", "d": "de", "f": "ef", "g": "ge", "h": "ha", "j": "jot", "k": "ka", "l": "el",
            "m": "em", "n": "en", "p": "pe", "q": "ku", "r": "er", "s": "es", "t": "te", "v": "vau", "w": "we",
            "x": "ix", "z": "zett", "ß": "es"}


def normalize(text: str, fold: bool = True) -> str:
    """Lowercase, strip punctuation and collapse whitespace; keeps umlauts.

    With ``fold`` (for scoring), spelling variants that sound the same are
    merged: ß -> ss (Whisper often writes the Swiss "Grosses" for "Großes"),
    and a lone letter becomes its name ("S" -> "es"). ``fold=False`` keeps the
    old form for split hashing, so no clip changes split.

    Digits are spelled out ("12. Oktober" -> "zwölften oktober"), so Whisper's
    numeric output matches references written as spoken words.

    Hyphenated compounds are split ("Hin- und Rückflug" -> "hin und rückflug") so
    the reference and hypothesis agree regardless of hyphenation style.
    """
    text = spell_numbers(unicodedata.normalize("NFC", text)).lower()
    text = text.replace("-", " ")
    words = _NON_WORD.sub(" ", text).split()
    if fold:
        words = [_LETTERS.get(w, w).replace("ß", "ss") for w in words]
    return " ".join(words)
