"""Transcript normalisation shared by scoring and data splitting."""

from __future__ import annotations

import re
import unicodedata

from irregular_voice_google.numbers import spell_numbers

_NON_WORD = re.compile(r"[^\w\s]")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace; keeps umlauts and ß.

    Digits are spelled out ("12. Oktober" -> "zwölften oktober"), so Whisper's
    numeric output matches references written as spoken words.

    Hyphenated compounds are split ("Hin- und Rückflug" -> "hin und rückflug") so
    the reference and hypothesis agree regardless of hyphenation style.
    """
    text = spell_numbers(unicodedata.normalize("NFC", text)).lower()
    text = text.replace("-", " ")
    text = _NON_WORD.sub(" ", text)
    return " ".join(text.split())
