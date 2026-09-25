# irregular-voice-google

Personalised German speech recognition for a speaker with dysarthria, aimed at
task-oriented use (e.g. booking a flight) on a phone or laptop.

Plan: baseline German Whisper models → domain lexicon biasing → per-speaker
LoRA fine-tune → LLM slot-filling with spoken confirmation → retrain on
confirmed/corrected utterances.

**Overview for readers:** [docs/PITCH.md](docs/PITCH.md): results, every
algorithm used, the proposed learning loop, related work and references.

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
Spellings that sound the same count as equal: ß = ss (Whisper often writes the
Swiss "Grosses"), and a lone letter is scored as its name ("scharfes ß" and
"scharfes S" are both "scharfes es"). The train/dev/test split hash uses the
unfolded text, so no clip changed split. With this, the adapter's test WER is
13.2% (was 14.9%) and the base model's 55.4% (was 57.9%).

## Speaker profile and prompts

A profile (`data/speakers/<id>/profile.json`, git-ignored; format in
`resources/speakers/example/profile.json`) holds a speaker's adapter,
preprocessing, a few of their own phrases and personal terms. `--profile`
turns it into Whisper's prompt; Whisper reads a prompt as the preceding
transcript, not as instructions, so it holds phrases rather than a description.

```bash
uv run ivg-eval --manifest data/processed/trim/manifest.csv --split dev \
  --profile data/speakers/<id>/profile.json --prompt-parts phrases
```

`--prompt-parts` is `phrases` (default), `terms` (cities + airlines), both, or
`none`. On our dev set a few of the speaker's own phrases helped; a long term
list hurt. The prompt is capped at 200 tokens (Whisper's limit is 224) and may
not contain an evaluated sentence.

### Per-question prompts and grammars

`resources/questions_de.json` lists the booking questions (destination,
origin, date, airline, confirm). The app knows what it just asked, so each
answer gets a short prompt (speaker phrases, the likely values, the question)
and, on whisper.cpp, a GBNF grammar of `[carrier words] value [carrier words]`
that steers decoding to the listed values (`questions.py`):

```bash
uv run ivg-eval --model models/ggml/<model>.bin --question destination ...
```

The grammar is soft: an off-list or half-finished answer can still come out,
so the app should accept only answers where `questions.match()` finds a value
and the lowest token probability (`min_p` column) is high enough, and ask again
otherwise.

`questions.resolve(question, text, min_p)` makes that call:

- **accept**: a listed value is spelled out and `min_p` ≥ 0.5;
- **confirm** ("Meinten Sie Sieben?"): a listed value heard with low
  confidence, or a transcript that only *sounds like* one value — compared by
  Kölner Phonetik code (`phonetic.py`), so "Seben" → "sieben" and
  "Aushalten" → "ausschalten";
- **repeat**: no value, several sound-alikes, or a loop-guard flag.

Replaying the adapter's test transcripts of the command clips (15, against the
speaker's 25-command list) gave 12 accepted (all right), 2 confirmed (both
right) and 1 repeat; nothing wrong was accepted.

## Live demo

```bash
uv run ivg-demo --list-mics                       # find the headset's device index
uv run ivg-demo --profile data/speakers/<id>/profile.json \
  --model models/ggml/<adapter>-q5_0.bin --mic <index> --say
```

Asks the booking questions (spoken with `--say`), records each answer (Enter to
start, Enter to stop), transcribes it on the Mac with whisper.cpp and shows
what `resolve` decides: accept, "Meinten Sie …?" or ask again (three tries).
It ends by reading back the booking for a yes/no. Takes and a `session.json`
log go to `data/demo/<timestamp>/` (git-ignored).

Decoding is plain by default. The adapter was trained and scored without a
prompt or grammar, and with the grammar on it misheard clear answers
("Von München" → "Zurück", confidence below the threshold); `--prompt` and
`--grammar` turn them back on. `--wav a.wav b.wav …` replays files instead of
the microphone, for a dry run.

### Web version

```bash
uv run ivg-demo --web --profile data/speakers/<id>/profile.json \
  --model models/ggml/<adapter>-q5_0.bin
```

Opens http://localhost:8766 (bound to 127.0.0.1, so only this Mac can reach it).
The dialogue is the same as in the terminal (`Dialog` in `demo.py` is shared by
both). The browser speaks each question with its German voice, and a big mic
button (or the space bar) records the answer. The page shows what Whisper
heard, the decision (understood, "Meinten Sie …?" or again) and the booking
filling in. Takes and `session.json` go to `data/demo/<timestamp>/` as before.

When an answer is not a clear value, the page offers up to three candidates:
sound-alikes by Kölner Phonetik and near misses ranked by the voicing-aware
edit distance from `snap.py` (`questions.suggest`: "Bärlin" → Berlin,
"Modien" → morgen). The carrier words ("von", "nach", "bitte") are dropped
first, so "von" cannot sound like "Wien". Nothing is booked until the speaker
picks one. He can tap a big button, say the name, say its number ("eins",
"zwei", "drei"), or say "ja" when there is only one candidate. "Nein" or
"keins davon" asks the question again. The readback at the end has "Ja, buchen"
and "Nein" buttons as well.

For a phone on USB, `scripts/phone-bridge.sh` keeps `adb reverse` up: it
restores the bridge whenever the phone reconnects. The page stays usable
through hiccups:
- It keeps the screen on (Wake Lock), because a locked phone silences the
  microphone.
- It catches an all-silent recording, reconnects the microphone and asks again.
  The server also treats a silent take as "please repeat".
- It retries requests while the cable is out.
- It resumes the conversation after a reload (`/api/state`).

On a phone, the Mac still does the transcribing: `--web --lan` serves the
page over HTTPS on the local network (self-signed certificate, random token in
the printed URL); open it on a phone on the same Wi-Fi and accept the warning.
An Android emulator on the same Mac needs no HTTPS: run
`adb reverse tcp:8766 tcp:8766` and open http://localhost:8766 in its Chrome.

```bash
uv run ivg-demo --compare --model models/ggml/<adapter>-q5_0.bin
```

`--compare` shows the before/after instead: it transcribes the speaker's test
clips (never trained on) with the base model and the adapter and opens a local
page with each clip's waveform, spectrogram (0–8 kHz, which shows the
muffling) and playback, the two transcripts with wrong words marked, and the
overall WER. The page embeds the audio, so it is written to `data/demo/` and
must stay there.

### Home control (`--scenario home`)

Booking a flight takes four answers and a readback. Home control is simpler:
one sentence does something, and the page shows it at once.

```bash
uv run ivg-demo --web --scenario home --profile data/speakers/<name>/profile.json --model models/ggml/<adapter>-q5_0.bin
```

A command has three parts: device (Licht, Lampe, Heizung, Rollo, Musik,
Fernseher), room and action. `resources/home_de.json` lists each part's values
and synonyms, e.g. "dunkel" means "aus" and "Jalousie" means "Rollo".
- One sentence can hold several commands. "Licht im Flur an, Büro dunkel"
  switches two lights; a part left out is taken from the command before it.
- Only words that are spelled out count, so an unrelated sentence does nothing
  and is asked again.
- A missing part is asked for on its own. "Rollo runter" gets "In welchem
  Raum?". A missing action is offered as numbered choices that fit the device
  ("eins: an, oder zwei: aus?").
- In those one-word answers, sound-alikes and near misses count, as in the
  flight dialog.
- If Whisper is unsure of a sentence, it is read back first.
- Each room shows its devices' last state.
- The session log is saved after every answer.

`resources/prompts/home_de.txt` is the matching recording list: commands,
several commands in one breath, and the single words asked for when a part is
missing. Record it with `uv run ivg-record --prompts resources/prompts/home_de.txt`.

### Snapping non-words to real words

The adapter often hears the right sounds but writes a non-word: "geklabt" for
"geklappt". `snap.py` replaces any word that the German frequency list
([wordfreq](https://pypi.org/project/wordfreq/)) has never seen, and that is
not in the speaker's train transcripts or the lexicons, with a real word that
has the same Kölner Phonetik code and is at most one or two edits away.
Voicing swaps (b/p, d/t, g/k), doubled consonants and h count as half an edit.
Among equally close words, the more frequent one wins. Rare real words
("pufft") are left alone.

Real words that are wrong ("Aushalten" for "Ausschalten") are only fixed for
short answers to a command prompt. `resources/commands_de.txt` lists the app's
commands. An answer that is not a command, but has exactly one command's sound
code and the same number of words, becomes that command. Longer sentences
never match, and two commands with the same code are never snapped.

```bash
uv run ivg-eval --model models/ggml/<adapter>-q5_0.bin --split test --snap ...
uv run ivg-demo --compare --snap --model models/ggml/<adapter>-q5_0.bin
```

| Split | Adapter WER | + snap WER |
| --- | --- | --- |
| dev (35 clips) | 13.9% | 10.9% |
| test (39 clips) | 13.2% | 10.7% |

On test it fixed geklabt, Seben → Sieben and Aushalten → Ausschalten.
No correct word was changed.
Some snaps turn one wrong word into another ("Zinge" becomes "Singe" when he
said "Ziege"). So the snapped text is for display and free text only, never
for booking values. Booking values go through `questions.resolve`.

#### Experiment: a small local language model (didn't help)

`llmfix.py` lets a local Ollama model propose a corrected sentence for real
words that make no sense ("Schweich schmeckt seidig"). A changed word is kept
only if it is a real word, sounds close (Kölner Phonetik code at most 1–2
edits away) and replaces exactly one heard word:

```bash
ollama serve & ollama pull qwen2.5:3b
uv run ivg-eval --model models/ggml/<adapter>-q5_0.bin --split dev --snap --llm qwen2.5:3b ...
```

With `qwen2.5:3b` on dev, WER went from 10.9% (snap) to 15.3%, and CER from
3.4% to 5.9%. It made one right fix (sackt → sägt) and broke eight correct
words (Zaun → Zun, warm → wär, Thymian → Thunfisch, Jara → Joghurt, …).
Tightening the sound check would not have stopped most of these: the model
has too little sense of German for sentences that are odd on purpose. It needs
a stronger model, and a gate that scores candidates instead of trusting a
rewrite.

## whisper.cpp

The app runtime is [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
(`brew install whisper-cpp`): fast on a Mac's GPU, small quantized models, no
Python, per-token probabilities and grammar-constrained decoding. The prompt
limit is the same as in Python (half the 448-token context).

```bash
uv run ivg-ggml --model primeline/whisper-large-v3-turbo-german --quantize q5_0
uv run ivg-ggml --model models/<adapter-dir> --quantize q5_0   # merges the LoRA first
uv run ivg-eval --model models/ggml/primeline__whisper-large-v3-turbo-german-q5_0.bin ...
```

`ivg-eval` runs any `--model` ending in `.bin` through `whisper-cli`. On our
dev set q5_0 (0.57 GB) scored the same as f16 (1.6 GB).

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

On a 24 GB GPU, `--batch-size 4 --grad-accum 2 --grad-checkpointing` fits in
about 7 GB (the default batch of 8 without checkpointing runs out of memory).
Progress is printed every `--log-every` batches with loss and time left, so a
run can be followed with `tail -f` on its log. Each epoch also reports dev CER
and the WER on a fixed sample of the speaker's own train clips (unaugmented,
same size as dev); a train WER far below dev WER means the adapter is
memorising the recordings.

Two ways to stretch a small recording set:

```bash
uv run ivg-train --augment 0.5                    # speed, muffle, reverb, noise, gain on the fly
uv run ivg-synth --exclude data/processed/trim/manifest.csv   # macOS: TTS booking answers
uv run ivg-train --augment 0.5 --extra-manifest data/synthetic/booking/manifest.csv
```

`--dora` trains DoRA instead of plain LoRA and `--rank` sets the adapter rank
(default 32).

- `--augment P` applies each augmentation with probability `P` to training
  batches only (numpy, no ffmpeg needed on the GPU box). Augmented runs need
  more epochs; use a long `--patience` so a lucky early epoch does not end the run.
- `ivg-synth` renders the answers to the booking questions (cities, dates,
  airlines, whole requests) with the German macOS `say` voices, slowed and
  low-passed, into `data/synthetic/booking/` (git-ignored). Sentences that
  appear in the patient's dev/test split are skipped, and `ivg-train` checks
  extra manifests for leaks too; extra data is always training-only.

Patient audio should not go to a cloud GPU unless the consent covers it.

### Other speakers: the adapter is personal

The adapter is trained on one speaker. To see whether it carries over, the
test split of
[B-Czarnetzki/dysarthric_german](https://huggingface.co/datasets/B-Czarnetzki/dysarthric_german)
was scored as well: 125 clips (38 min) of other German speakers with
dysarthria. Its card names no source, license or speakers, so it is used for
internal comparison only and stays under the git-ignored `data/external/`.

```bash
uvx --from huggingface_hub hf download B-Czarnetzki/dysarthric_german \
    data/test-00000-of-00001-b8a965f992d610f9.parquet --repo-type dataset \
    --local-dir data/external/dysarthric_german
uv run --with pyarrow --with soundfile python scripts/external_dysarthric_german.py
uv run ivg-eval --manifest data/external/dysarthric_german/manifest.csv \
    --model models/ggml/primeline__whisper-large-v3-turbo-german-q5_0.bin \
    --model models/ggml/<adapter>-q5_0.bin
```

| test WER | his clips (39) | other speakers (125) |
|---|---|---|
| base `primeline/whisper-large-v3-turbo-german` (no prompt, no snap) | 55.4% | 60.6% |
| his adapter (no prompt, no snap) | 13.2% | 109.4% (12 loops) |
| his adapter + his profile prompt + snap (the demo setup) | 10.7% (snap only) | 113.3% (9 loops) |

The adapter cuts his errors by three quarters and does not transfer: it is
better than the base model on 7 of the 125 other clips and worse on 108. It
learned his recordings, not dysarthric German in general. Much of his material
is lists of short words, so it splits other people's speech into syllables
("Auch, alle, Mö, W, Mö, B, Jüngel, U." for "Auch alle Menschen mit
Behinderungen.") and fills in words from his sentences. Each patient needs
their own adapter. The base model's score on other speakers (60.6%, median
59%, from 8 clips under 20% to 23 over 100%) is the fair reference for how
hard dysarthric German is without adaptation.

His profile prompt and snap do not help either: the prompt holds his phrases,
not theirs, and snap cannot rejoin speech split into syllables. The adapter is
also very unsure on these clips (lowest word probability about 0.05–0.09), so
in the demo their takes would be read back or asked again, not acted on.

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
  augment.py     # training-time audio augmentation
  synth.py       # synthetic TTS booking utterances
  preprocess.py  # trim / tempo / EQ variants
  guard.py       # Whisper repetition-loop guard
  profile.py     # per-speaker profile -> Whisper prompt
  questions.py   # per-question prompts, GBNF grammars, answer matching/decisions
  phonetic.py    # Kölner Phonetik sound codes
  snap.py        # non-words -> same-sounding real words (wordfreq)
  llmfix.py      # experiment: local LLM word fixes, gated by sound
  ggml.py        # HF model / LoRA adapter -> whisper.cpp ggml
  cpp.py         # whisper.cpp backend (whisper-cli)
  demo.py        # live booking demo (terminal or --web: demo.html) + base-vs-adapter page (compare.html)
  home.py        # home-control dialog for the demo (--scenario home)
resources/
  lexicon/       # one <slot>.txt per slot
  questions_de.json  # booking questions: slots and carrier words
  home_de.json   # home control: devices, rooms, actions and their synonyms
  speakers/example/  # profile format (real profiles live in data/speakers/)
  prompts/       # German recording prompts + lexicon story
scripts/
  phone-bridge.sh  # keeps adb reverse up for the phone demo
  external_dysarthric_german.py  # other speakers' test clips -> manifest
tests/
```
