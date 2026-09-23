"""Per-question context: what the app just asked decides the prompt and the grammar.

``resources/questions_de.json`` lists the booking questions. Each answer is
``[before] value [after]``: the value comes from lexicon slots (``slots``)
or an explicit ``values`` list; ``before``/``after`` are the carrier words
people wrap around it ("nach Berlin", "am 12. März bitte").

- ``prompt(question, profile)``: a few of the speaker's phrases, the likely
  values, then the question itself (Whisper reads the prompt as the preceding
  transcript, so the question goes last).
- ``grammar(question)``: a GBNF grammar for whisper.cpp's ``--grammar``. The
  constraint is soft (``--grammar-penalty``), so an off-list answer can still
  come through, but it is strongly steered to the list.
- ``match(question, text)``: the value the answer names, or None.
- ``resolve(question, text, min_p)``: what the app should do with an answer:
  accept it, ask "Meinten Sie …?", or ask again.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from irregular_voice_google import phonetic
from irregular_voice_google.lexicon import load_lexicon
from irregular_voice_google.profile import Profile
from irregular_voice_google.text import normalize

DAY_RULE = '("3" [01] | [12] [0-9] | [1-9]) "."'


@dataclass
class Question:
    name: str
    ask: str
    values: list[str]
    before: list[str] = field(default_factory=list)
    after: list[str] = field(default_factory=list)
    day_numbers: bool = False


def load_questions(path: str | Path = "resources/questions_de.json",
                   lexicon: str | Path = "resources/lexicon") -> dict[str, Question]:
    slots = load_lexicon(lexicon, raw=True)
    questions = {}
    for name, spec in json.loads(Path(path).read_text(encoding="utf-8")).items():
        values = [*spec.get("values", []), *(t for slot in spec.get("slots", []) for t in slots[slot])]
        if not values:
            raise ValueError(f"question {name!r} has no values")
        questions[name] = Question(name, spec["ask"], list(dict.fromkeys(values)), spec.get("before", []),
                                   spec.get("after", []), spec.get("day_numbers", False))
    return questions


def prompt(question: Question, profile: Profile | None = None, n_phrases: int = 3) -> str:
    phrases = " ".join(profile.prompt_phrases[:n_phrases]) if profile else ""
    return " ".join(p for p in (phrases, ", ".join(question.values) + ".", question.ask) if p)


def _alternatives(options: list[str]) -> str:
    variants = dict.fromkeys(v for o in options for v in (o, o[:1].upper() + o[1:]))  # sentence-initial caps
    return " | ".join('"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"' for v in variants)


def grammar(question: Question) -> str:
    value = "word"
    rules = [f"word ::= {_alternatives(question.values)}"]
    if question.day_numbers:
        rules.append(f"day ::= {DAY_RULE}")
        value = '(day " " word | day | word)'
    head = f'(before " ")? {value}' if question.before else value
    tail = ' (" " after)?' if question.after else ""
    if question.before:
        rules.append(f"before ::= {_alternatives(question.before)}")
    if question.after:
        rules.append(f"after ::= {_alternatives(question.after)}")
    return "\n".join([f'root ::= " " {head}{tail} [.!?]?', *rules]) + "\n"


def match(question: Question, text: str) -> str | None:
    """The longest listed value named in ``text`` (so "Frankfurt am Main" beats "Frankfurt")."""
    norm = f" {normalize(text)} "
    found = [v for v in question.values if re.search(rf"(?<!\w){re.escape(normalize(v))}(?!\w)", norm)]
    return max(found, key=len) if found else None


def sounds_like(question: Question, text: str) -> list[str]:
    """Listed values whose Kölner Phonetik code appears in ``text`` ("Seben" -> "sieben")."""
    heard = f" {phonetic.code(normalize(text))} "
    found = [v for v in question.values if (c := phonetic.code(normalize(v))) and f" {c} " in heard]
    return [v for v in found if not any(v != w and normalize(v) in normalize(w) for w in found)]  # longest only


@dataclass
class Decision:
    action: str  # "accept", "confirm" ("Meinten Sie …?") or "repeat"
    value: str | None = None
    candidates: list[str] = field(default_factory=list)


def resolve(question: Question, text: str, min_p: float | None = None, threshold: float = 0.5,
            flagged: bool = False) -> Decision:
    """Accept only a spelled-out value heard confidently; a low-confidence or sound-alike value is confirmed.

    ``min_p`` is whisper.cpp's lowest token probability (None skips the check);
    ``flagged`` is the loop guard's verdict, which always means "please repeat".
    """
    if flagged:
        return Decision("repeat")
    if value := match(question, text):
        return Decision("accept" if min_p is None or min_p >= threshold else "confirm", value, [value])
    candidates = sounds_like(question, text)
    if len(candidates) == 1:
        return Decision("confirm", candidates[0], candidates)
    return Decision("repeat", None, candidates)
