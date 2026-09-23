"""Guard against Whisper repetition loops ("Jürgen, Jürgen, Jürgen, …").

Two layers:

- ``max_new_tokens(seconds)`` caps generation by clip length, so a loop cannot
  run to Whisper's 448-token limit.
- ``guard(text, seconds)`` collapses runaway repeats and says why it flagged
  the output. In the booking app a flagged transcript means "ask the speaker
  to repeat", never "act on it".
"""

from __future__ import annotations

from irregular_voice_google.text import normalize

TOKENS_PER_SECOND = 8  # generous: fluent German is ~5 tokens/s and he speaks slowly
MAX_WORDS_PER_SECOND = 6
# A unigram may legitimately repeat a few times ("null null null"); longer n-grams rarely do.
MIN_REPEATS = {1: 4, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3}


def max_new_tokens(seconds: float) -> int:
    return min(440, int(seconds * TOKENS_PER_SECOND) + 16)


def _collapse(words: list[str]) -> tuple[list[str], bool]:
    keys = [normalize(w) for w in words]
    for n, min_repeats in MIN_REPEATS.items():
        i = 0
        while i + n <= len(words):
            reps = 1
            while keys[i + reps * n : i + (reps + 1) * n] == keys[i : i + n]:
                reps += 1
            if reps >= min_repeats and any(keys[i : i + n]):
                del words[i + n : i + reps * n], keys[i + n : i + reps * n]
                return words, True
            i += 1
    return words, False


def guard(text: str, seconds: float | None = None) -> tuple[str, str | None]:
    """Return (cleaned text, reason or None). Reasons: ``repeat``, ``too_fast``."""
    words, reason = text.split(), None
    while True:
        words, changed = _collapse(words)
        if not changed:
            break
        reason = "repeat"
    cleaned = " ".join(words)
    if reason is None and seconds and len(normalize(cleaned).split()) > MAX_WORDS_PER_SECOND * max(seconds, 0.5):
        reason = "too_fast"
    return cleaned, reason
