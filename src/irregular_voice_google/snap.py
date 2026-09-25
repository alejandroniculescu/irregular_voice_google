"""Snap non-words to the closest real German word that sounds the same.

The adapter usually hears the right sounds but can pick a spelling that is not
a word: "geklabt" for "geklappt" (the b/p voicing contrast is lost). Both have
the Kölner Phonetik code 4512. A word that the German frequency list
(``wordfreq``) has never seen and that is not in the speaker's own vocabulary
is replaced by the same-code word closest in spelling, and only if that is at
most one or two edits away. Voicing swaps (b/p, d/t, g/k), doubled consonants
and h count half, since that is how this speaker's errors look; among equally
close words the more frequent one wins ("geklabt" is as close to "geklebt" as
to "geklappt", and "geklappt" is more common). Rare real words
("pufft", "blechern") are seen, just rarely, so they stay. Real words are
never touched, so a real-word slip ("Aushalten" for "Ausschalten") stays; that
is what the per-question value lists and ``questions.resolve`` are for.

Real words get one exception: short answers to a command prompt. With a
``commands`` list, an utterance that is not a command but has exactly one
command's sound code and word count becomes that command ("Aushalten" ->
"Ausschalten", both 08526). The app knows it asked for a command, the same
knowledge ``questions.resolve`` uses.

A snapped word can still be wrong ("Zinge" -> "Zunge" for "Ziege"): it trades
an obvious non-word for a plausible word, so snapped text is for display and
free text, not for booking values.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path

from wordfreq import top_n_list, zipf_frequency

from irregular_voice_google import phonetic
from irregular_voice_google.manifest import load_manifest

MIN_ZIPF = 2.0  # replacement candidates must be at least this common (Bügelbrett is 2.3)
RARE_ZIPF = 1.2  # a "real" word rarer than this counts as a non-word (rücksichtloser 1.1; pufft 1.4 stays)
WORD = re.compile(r"[A-Za-zÄÖÜäöüß]+")


# Consonants that differ only in voicing (or not at all in sound): the speaker's typical confusion.
_PAIRS = [set("bp"), set("dt"), set("gkq"), set("fvw"), set("sz")]
_UMLAUTS = [set("aä"), set("oö"), set("uü"), set("eä")]
_VOWELS = "aeiouäöüy"


def _swap(x: str, y: str, variants: bool) -> float:
    pairs = _PAIRS + _UMLAUTS if variants else _PAIRS
    return 0.0 if x == y else 0.5 if any(x in p and y in p for p in pairs) else 1.0


def _gap(w: str, i: int, variants: bool) -> float:
    """Cost of inserting/deleting w[i]: half for a doubled consonant or a silent h.

    ``variants`` also halves sounds that German spelling hides or that speech often softens: a
    doubled vowel (Kaffe/Kaffee), the e of ie, an r after a vowel
    (vocalized: Ferse/Fehse), an s between consonants or starting a cluster
    (stramme/tramme, rücksichtsloser/rücksichtloser), the f of a final pf (Topf/Top),
    the p of an initial pf (Pfeffer/Feffer), the d of dj (Djamila/Jamila) and the t of tz.
    """
    c, prev, nxt = w[i], w[i - 1] if i else "", w[i + 1 : i + 2]
    if c == "h" or (c not in _VOWELS and c in (prev, nxt)):
        return 0.5
    if not variants:
        return 1.0
    cons = lambda x: bool(x) and x not in _VOWELS  # noqa: E731
    return 0.5 if (c in (prev, nxt) or (c == "e" and prev == "i")
                   or (c == "r" and prev in _VOWELS and prev and not (nxt and nxt in _VOWELS))
                   or (c == "s" and cons(nxt) and (not prev or cons(prev)))
                   or (c == "f" and prev == "p" and i == len(w) - 1) or (c == "p" and i == 0 and nxt == "f")
                   or (c == "d" and nxt == "j") or (c == "t" and nxt == "z")) else 1.0


def edits(a: str, b: str, variants: bool = False) -> float:
    """Edit distance where voicing swaps (b/p, d/t, g/k), doubled consonants and h cost half.

    ``variants`` (for snapping free text) also halves umlaut swaps (ä/a) and the
    hidden sounds in ``_gap``. Booking values (``questions.suggest``) keep the
    strict costs: there a looser match lets noise ("Jungen") reach a value ("München")."""
    prev = [0.0]
    for j in range(len(b)):
        prev.append(prev[-1] + _gap(b, j, variants))
    for i, ca in enumerate(a):
        cur = [prev[0] + _gap(a, i, variants)]
        for j, cb in enumerate(b):
            cur.append(min(prev[j + 1] + _gap(a, i, variants), cur[j] + _gap(b, j, variants),
                           prev[j] + _swap(ca, cb, variants)))
        prev = cur
    return prev[-1]


class Snapper:
    def __init__(self, vocabulary: Iterable[str] = (), min_zipf: float = MIN_ZIPF, n: int = 300_000,
                 commands: Iterable[str] = ()):
        """``vocabulary``: texts whose words always count as real (speaker's train transcripts, lexicons).
        ``commands``: short phrases a sound-alike answer is snapped to as a whole."""
        by_command: dict[str, list[str]] = {}
        for c in commands:
            by_command.setdefault(phonetic.code(c), []).append(c)
        self.commands = {c.lower() for c in commands}
        self.by_command = {k: v[0] for k, v in by_command.items() if len(v) == 1}  # ambiguous codes: never snap
        self.min_zipf = min_zipf
        self.vocabulary = {w.lower() for text in vocabulary for w in WORD.findall(text)}
        self.freq: dict[str, float] = {}
        for w in top_n_list("de", n):
            if not WORD.fullmatch(w):
                continue
            if (z := zipf_frequency(w, "de")) < min_zipf:
                break  # the list is sorted by frequency
            self.freq[w] = z
        self.by_code: dict[str, set[str]] = {}
        for w in {*self.freq, *self.vocabulary}:
            for k in phonetic.variants(w):
                self.by_code.setdefault(k, set()).add(w)

    def known(self, word: str) -> bool:
        w = word.lower()
        return w in self.vocabulary or w in self.freq or zipf_frequency(w, "de") > 0

    def candidates(self, word: str) -> list[tuple[float, float, str]]:
        """(edits, -zipf, word) for every other real word that shares one of ``word``'s sound codes."""
        w = word.lower()
        found = {c for k in phonetic.variants(w) for c in self.by_code.get(k, ())} - {w}
        return sorted((edits(w, c, variants=True), -self.freq.get(c, self.min_zipf), c) for c in found)

    def word(self, word: str) -> str:
        w = word.lower()
        if len(w) < 3 or w in self.vocabulary:
            return word
        if zipf_frequency(w, "de") >= RARE_ZIPF:  # a real word (a more common twin is no reason: Kelle/Keller)
            return word
        limit = 1 if len(w) < 6 else 2
        scored = [s for s in self.candidates(w) if s[0] <= limit]
        if not scored:
            return word
        best = min(scored)[2]
        return best[:1].upper() + best[1:] if word[:1].isupper() else best

    def command(self, text: str) -> str | None:
        """The command ``text`` sounds exactly like, if it is not already one."""
        words = WORD.findall(text)
        if not words or " ".join(words).lower() in self.commands:
            return None
        c = self.by_command.get(phonetic.code(text))
        return c if c and len(c.split()) == len(words) else None

    def text(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)  # "ö" as o + ¨ would split the word
        if c := self.command(text):
            return c + text[len(text.rstrip(".!?")):]  # keep the final punctuation
        return WORD.sub(lambda m: self.word(m[0]), text)


def for_speaker(manifest: str | Path, lexicon: str | Path = "resources/lexicon",
                commands: str | Path = "resources/commands_de.txt") -> Snapper:
    """Snapper that also knows the speaker's train transcripts, the slot lexicons and the app's commands
    (never dev/test transcripts)."""
    texts = [u.text for u in load_manifest(manifest) if u.split == "train"]
    lines = [line.strip() for line in Path(commands).read_text(encoding="utf-8").splitlines()]
    cmds = [line for line in lines if line and not line.startswith("#")]
    return Snapper(texts + cmds + [p.read_text(encoding="utf-8") for p in sorted(Path(lexicon).glob("*.txt"))],
                   commands=cmds)
