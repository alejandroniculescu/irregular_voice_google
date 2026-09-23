"""Per-speaker profile: everything the app needs to recognise one speaker.

Profiles name a real person, so they live in the git-ignored
``data/speakers/<id>/profile.json``; ``resources/speakers/example/`` shows the
format. Fields (all but ``speaker`` optional):

- ``adapter``        LoRA adapter directory from ``ivg-train``
- ``preprocess``     preprocessing steps, e.g. ``"trim"`` (see ``preprocess.py``)
- ``prompt_phrases`` sentences in the speaker's own vocabulary and style
- ``lexicon``        slot lexicon directory; ``extra_terms`` adds personal words

Whisper does not follow instructions: its prompt is read as *preceding
transcript*, so it holds example phrases and terms, not a description.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from irregular_voice_google.lexicon import load_lexicon
from irregular_voice_google.manifest import Utterance
from irregular_voice_google.text import normalize

PROMPT_PARTS = ("phrases", "terms")


@dataclass
class Profile:
    speaker: str
    adapter: str | None = None
    preprocess: str = ""
    prompt_phrases: list[str] = field(default_factory=list)
    lexicon: str | None = None
    extra_terms: list[str] = field(default_factory=list)


def load_profile(path: str | Path) -> Profile:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(data) - set(Profile.__dataclass_fields__)
    if unknown:
        raise ValueError(f"{path}: unknown profile fields {sorted(unknown)}")
    return Profile(**data)


def terms(profile: Profile) -> list[str]:
    lexicon = load_lexicon(profile.lexicon) if profile.lexicon and Path(profile.lexicon).is_dir() else {}
    # Only proper nouns help as bias; function words ("ja", "Uhr") are already easy for Whisper.
    slots = [t for slot in ("cities", "airlines") for t in lexicon.get(slot, [])]
    return list(dict.fromkeys([*profile.extra_terms, *(t.title() for t in slots)]))


def prompt_text(profile: Profile, parts: str | list[str] = PROMPT_PARTS) -> str:
    """Prompt built from the chosen parts, most specific (personal phrases) last."""
    parts = [p for p in (parts.split(",") if isinstance(parts, str) else parts) if p and p != "none"]
    if set(parts) - set(PROMPT_PARTS):
        raise ValueError(f"prompt parts must be from {PROMPT_PARTS}")
    chunks = []
    if "terms" in parts:
        chunks.append(", ".join(terms(profile)) + ".")
    if "phrases" in parts:
        chunks.append(" ".join(profile.prompt_phrases))
    return " ".join(c for c in chunks if c.strip(" ."))


def check_prompt_not_in(prompt: str, utterances: list[Utterance]) -> None:
    """Refuse a prompt that contains a sentence being evaluated — that would leak the answer."""
    norm = f" {normalize(prompt)} "
    leaked = [u.text for u in utterances if len(normalize(u.text).split()) > 1 and f" {normalize(u.text)} " in norm]
    if leaked:
        raise SystemExit(f"prompt contains {len(leaked)} evaluated sentence(s), e.g. {leaked[0]!r}")
