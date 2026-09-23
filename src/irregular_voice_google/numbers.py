"""Spell out digits as German words so "Am 12. Oktober" scores like "am zwölften Oktober"."""

from __future__ import annotations

import re

_ONES = ["null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun", "zehn",
         "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn"]
_TENS = ["", "", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig"]
_ORDINAL_STEMS = {1: "erst", 3: "dritt", 7: "sieb", 8: "acht"}
# "12." followed by a word is an ordinal ("am 12. Oktober"); at the end of a sentence it is not.
_ORDINAL = re.compile(r"\b(\d{1,4})\.(?=\s+\w)")
_CARDINAL = re.compile(r"\d+")


def cardinal(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        ones, tens = n % 10, n // 10
        return (("ein" if ones == 1 else _ONES[ones]) + "und" if ones else "") + _TENS[tens]
    if n < 1000:
        rest = n % 100
        return ("" if n // 100 == 1 else cardinal(n // 100)) + "hundert" + (cardinal(rest) if rest else "")
    if n < 1_000_000:
        rest = n % 1000
        head = "" if n // 1000 == 1 else cardinal(n // 1000)
        return head + "tausend" + (cardinal(rest) if rest else "")
    return " ".join(_ONES[int(d)] for d in str(n))  # phone/booking numbers: digit by digit


def ordinal(n: int) -> str:
    """Dative/accusative form ("am zwölften", "den ersten"), the usual one for dates."""
    stem = _ORDINAL_STEMS.get(n) or cardinal(n) + ("t" if n < 20 else "st")
    return stem + "en"


def spell_numbers(text: str) -> str:
    text = _ORDINAL.sub(lambda m: ordinal(int(m[1])), text)
    return _CARDINAL.sub(lambda m: f" {cardinal(int(m[0]))} ", text)
