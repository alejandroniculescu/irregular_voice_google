# irregular-voice-google

Personalised German speech recognition for a speaker with dysarthria, aimed at
task-oriented use (e.g. booking a flight) on a phone or laptop.

Plan: baseline German Whisper models → domain lexicon biasing → per-speaker
LoRA fine-tune → LLM slot-filling with spoken confirmation → retrain on
confirmed/corrected utterances.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and `ffmpeg` (for phone formats like `.m4a`).

```bash
uv sync
```

## Data

Patient audio lives in `data/`, results in `results/` and fine-tuned adapters
in `models/`; all are git-ignored
and must never be committed. Get written consent before recording.

### Recording

A local recording page shows one prompt from
`resources/prompts/booking_de.txt` at a time (fewest takes first), lets the
speaker listen back and re-record, and saves each take as a WAV + `.txt`
transcript under `data/raw/<speaker>/<date>/`. Audio never leaves the machine.

```bash
uv run ivg-record                 # on this laptop: http://localhost:8765
uv run ivg-record --lan           # on a phone on the same Wi-Fi (HTTPS + token URL)
```

With `--lan`, open the printed `https://…?token=…` link on the phone and
accept the self-signed certificate warning (browsers only allow the microphone
over HTTPS). Browser noise suppression and auto-gain are disabled so the
recordings keep the speaker's real voice characteristics. Aim for two takes per
prompt, spread over several short sessions on different days.

`resources/prompts/story_de.txt` is a short first-person story about booking a
trip that uses every lexicon term (cities, dates, airlines, booking words) in
51 short sentences — more natural to read than isolated prompts:

```bash
uv run ivg-record --prompts resources/prompts/story_de.txt
```

### Manifest

Recordings from `ivg-record` are already in the right layout. For other
audio, put each recording next to a same-named `.txt` transcript
(`clip_001.m4a` + `clip_001.txt`), then build the manifest:

```bash
uv run ivg-manifest data/raw --out data/manifest.csv
```

The manifest assigns `train`/`dev`/`test` splits from a hash of the transcript
(~70/10/20), so repeated recordings of the same sentence always share a split
and nothing leaks from train into test. An explicit `split` column overrides it.

### Importing a `voice_samples_*` export

```bash
uv run ivg-import-samples ~/Desktop/voice_samples_2026-09-23
```

Copies the audio under `data/raw/<export>/` and writes `data/manifest.csv` from
the export's phrase manifest (its splits are kept, `test_adapt` → `test`) and
the reviewed command/number segments (transcript-hash split).

### Preprocessing variants

```bash
uv run ivg-preprocess --steps trim                 # -> data/processed/trim/manifest.csv
uv run ivg-preprocess --steps trim,tempo=1.2,eq=6  # -> data/processed/trim_tempo1.2_eq6/
```

Steps (applied in order with ffmpeg): `trim` cuts leading/trailing silence,
`tempo=<x>` speeds up without changing pitch, `eq=<dB>` boosts 2–5 kHz. Texts
and splits are copied unchanged, so pass the variant's manifest to `ivg-eval`
or `ivg-train`. Whatever variant is used for training must also be applied to
live audio (`preprocess.load`).

## Baseline evaluation

```bash
uv run ivg-eval --manifest data/manifest.csv --split test
```

Defaults to `primeline/whisper-large-v3-turbo-german` and
`openai/whisper-large-v3-turbo`; pass `--model <hf-id>` (repeatable) to compare
others. Reports WER, CER and keyword recall per slot using the lexicons in
`resources/lexicon/` (cities, dates, airlines, booking words). Per-utterance
outputs go to `results/<timestamp>/`.

Whisper occasionally gets stuck repeating a word. A loop guard (`guard.py`)
caps generation length by clip duration and collapses runaway repeats; flagged
utterances are counted in the `loops` column and marked in the per-utterance
CSV. Training uses the same guard when scoring dev. In the app, a flagged
transcript should trigger "please repeat", never an action.

Scoring spells out digits before comparing ("Am 12. Oktober" → "am zwölften
oktober"), so references may use either digits or words. Ordinals use the
dative form common in dates; numeric-only dates like "12.10." are not expanded.

## Per-speaker fine-tuning (LoRA)

```bash
uv run ivg-train                                  # on a CUDA GPU (lab machine)
uv run ivg-eval --model models/<adapter-dir> --model primeline/whisper-large-v3-turbo-german
```

Trains LoRA adapters (encoder + decoder, SpecAugment) on the `train` split of
`primeline/whisper-large-v3-turbo-german`, scores `dev` after every epoch and
keeps the best adapter under `models/` (git-ignored — adapters are learned from
patient audio). It refuses to run if a train sentence also appears in dev/test,
and never reads the test split. On a Mac only smoke runs are practical:
`uv run ivg-train --base openai/whisper-tiny --limit 8 --epochs 2`.

Patient audio should not go to a cloud GPU unless the consent covers it.

## Tests

```bash
uv run pytest
```

## Layout

```
src/irregular_voice_google/
  text.py        # German transcript normalisation
  numbers.py     # digits -> German number words
  manifest.py    # manifest building/loading, leak-free splits
  lexicon.py     # slot lexicons and keyword recall
  evaluate.py    # baseline evaluation CLI
  recorder.py    # local recording server (+ recorder.html)
  import_samples.py  # import a voice_samples_* export
  train.py       # per-speaker LoRA fine-tune
  preprocess.py  # trim / tempo / EQ variants
  guard.py       # Whisper repetition-loop guard
resources/
  lexicon/       # one <slot>.txt per slot
  prompts/       # German recording prompts + lexicon story
tests/
```
