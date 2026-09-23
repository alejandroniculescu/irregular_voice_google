"""Experiment: a small local language model fixes misheard real words, gated by sound.

``snap.py`` only fixes non-words. "Schweiß schmeckt salzig" heard as "Schweich
schmeckt seidig" is all real words; only sense says "seidig" is wrong. A local
Ollama model (nothing leaves the machine) proposes a corrected sentence, and
each changed word is kept only if

- it is a real word (``snap.Snapper.known``),
- its Kölner Phonetik code is at most one edit from the heard word's code (two
  for codes of four digits or more): "Schweich" 834 -> "Schweiß" 838 passes,
  a free rewrite does not,
- and it replaces exactly one heard word (no inserted or dropped words).

Everything else stays as heard, so the model can pick a sound-alike but never
invent a sentence. It can still pick the wrong sound-alike, so like ``snap``
this is for display and free text, never for booking values.
"""

from __future__ import annotations

import difflib
import json
import re
import urllib.request

from irregular_voice_google import phonetic
from irregular_voice_google.snap import WORD, Snapper

OLLAMA = "http://127.0.0.1:11434/api/generate"
PROMPT = """Das ist die automatische Transkription eines Satzes, gesprochen von einer Person mit \
undeutlicher Aussprache. Einzelne Wörter können falsch erkannt sein: dann steht dort ein ähnlich \
klingendes, aber falsches Wort. Die Sätze sind oft absichtlich ungewöhnliche Sprechübungen \
(z. B. "Zischende Säge sägt Zapfen") und dürfen seltsam bleiben. Ersetze nur Wörter, die im \
Satz offensichtlich keinen Sinn ergeben, durch ein ähnlich klingendes Wort. Ändere nichts \
anderes, füge nichts hinzu, lass nichts weg. Antworte nur mit dem Satz.

Transkription: {text}
Satz:"""


def code_distance(a: str, b: str) -> int:
    """Levenshtein distance between two Kölner Phonetik codes."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def sounds_close(heard: str, proposed: str) -> bool:
    a, b = phonetic.code(heard), phonetic.code(proposed)
    return bool(a and b) and code_distance(a, b) <= (2 if max(len(a), len(b)) >= 4 else 1)


def ollama(text: str, model: str, url: str = OLLAMA, timeout: float = 120) -> str:
    body = json.dumps({"model": model, "prompt": PROMPT.format(text=text), "stream": False,
                       "options": {"temperature": 0, "num_predict": 80}}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())["response"].strip()
    return out.splitlines()[0].strip().strip('"„“') if out else ""


def gate(heard: str, proposed: str, snapper: Snapper) -> str:
    """``heard`` with only the proposed one-for-one word swaps that are real and sound close."""
    words = list(WORD.finditer(heard))
    new = WORD.findall(proposed)
    swaps: dict[int, str] = {}
    matcher = difflib.SequenceMatcher(a=[m[0].lower() for m in words], b=[w.lower() for w in new], autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op != "replace" or i2 - i1 != j2 - j1:
            continue
        for i, j in zip(range(i1, i2), range(j1, j2)):
            if snapper.known(new[j]) and sounds_close(words[i][0], new[j]):
                w = new[j]
                swaps[i] = w[:1].upper() + w[1:] if words[i][0][:1].isupper() else w[:1].lower() + w[1:]
    out, last = [], 0
    for i, m in enumerate(words):
        out += [heard[last:m.start()], swaps.get(i, m[0])]
        last = m.end()
    return "".join(out) + heard[last:]


def fix(text: str, snapper: Snapper, model: str, propose=ollama) -> str:
    """``text`` (already snapped) with the model's sound-alike word swaps that pass the gate."""
    if not re.search(r"\w", text):
        return text
    return gate(text, propose(text, model), snapper)
