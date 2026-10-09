# Open questions for Christian

Context: the home-assistant deployment, with the phone as a satellite doing edge compute, running the
personal LoRA adapter trained on Christian's voice.

## 2026-10-06

- **Phone as satellite.** The phone always on, listening and running the adapter locally (edge compute),
  sending only recognised commands to the home assistant? Or wake-word first, then local recognition?
  Battery, thermal and privacy expectations for an always-on phone.
- **More voice samples.** The adapter was trained on about 12 minutes; for home use we need more, and from
  the situations the satellite will actually hear: across the room, with the household around, tired,
  first thing in the morning. How many minutes, recorded how.
- **Tagging strategy.** For the new recordings: who tags, at what level (word, command, phone), with which
  tool, and how we check agreement between taggers; whether confirmed commands from daily use can become
  training data without hand tagging.
- **Training.** Where the adapter is retrained (laptop, server, never on the phone), how often, from which
  data, and how a new adapter is pushed to the satellite without breaking the running one.

## 2026-10-09 — to try: an on-device reply voice (NeuTTS, Neuphonic)

- **What it is.** NeuTTS-Air: text-to-speech, ~0.5 B parameters (small LLM + NeuCodec), GGUF through llama.cpp,
  runs on a phone or a Raspberry Pi, Apache 2.0, clones a voice from 3–15 s of clean reference audio, output
  watermarked. Repo: https://github.com/neuphonic/neutts-air . German is a separate "Nano" model under
  Neuphonic's own licence (NeuTTS Open License 1.0), to be read before anything ships.
- **Why here.** The satellite already runs whisper.cpp (GGML) for recognition; NeuTTS would give it a reply
  voice in the same on-device family, so the privacy line (nothing leaves the phone) holds for output as well
  as input.
- **Voice banking.** Cloning from a few seconds means the assistant could answer in Christian's own voice.
  Two questions for him: does he want that at all, and which reference, since a dysarthric reference clones
  the dysarthria; an older clean recording, or his current voice by choice. Consent is his to give for
  himself; for anyone else the product needs an explicit step.
- **What to measure if tried.** Latency to first audio on the phone (the README benchmarks a Galaxy A25);
  intelligibility of the German Nano voice on the command vocabulary (resources/commands_de.txt); whether
  the watermark survives the satellite's audio path (it should; it is the point).
- **Not for.** Nothing on the measurement side: synthetic speech is not ground truth for anything.

## 2026-10-09 — two experiments, written before any number is read

**E1. Does synthetic speech help the adapter, and does a cloned voice help more than generic voices?**
Held-out test: the same 39 recordings as the README table; metric WER with sound-based correction (and CER,
and WER before correction, reported). Arms, all trained with `--augment 0.5`, same base, rank, epochs, seed:
- A `lora-aug`: patient audio only (the README's 13.2 / 10.7 system);
- B `lora-aug-synth`: + 300 booking sentences in generic macOS German voices, slowed and low-passed (trained
  2026-09, never scored until now);
- C `lora-aug-neutts`: + the same sentences in the user's own voice cloned by NeuTTS (German Nano) from one
  4–14 s train-split clip (`ivg-synth-neutts`); no slowing, no low-pass, so the clone is tested as it is;
- D (if C helps): B + C together.
Reading: an arm "helps" if its WER on the 39 clips is lower than A's by more than the utterance-bootstrap 95 %
interval of the difference; "hurts" if higher by the same rule; otherwise no evidence either way. Prediction:
B no evidence (generic voices teach vocabulary, not his articulation); C hurts or no evidence (a codec clone
copies timbre, not the articulatory errors, so it teaches a voice that does not exist). If C helps, the
clone carries more of his speech than expected and voice banking becomes a data source, not only a feature.
Before C can run: the German NeuTTS model is gated on Hugging Face (accept its "NeuTTS Open License 1.0" with
the downloading account, text unavailable at the time of writing; read it first), and Christian's consent to
clone his voice for an internal training set, nothing published, clone deleted after the experiment unless he
wants it kept for voice banking.

**E2. A mixture of adapters (what "MoE with the LoRA" means here).** Not one adapter but several small ones,
one per condition the satellite will meet (morning, tired, across the room, with the household), and a gate
that picks or blends them per utterance from the audio itself. Needs the new recordings from the 2026-10-06
list first; the structure to train once they exist. Reading as E1: each condition's adapter against the single
adapter on that condition's held-out clips.

### E1, arms A and B scored 2026-10-09 (`results/20261009-132109/`, git-ignored; HF path, no prompt, GPU)

Same 39 clips, **121 reference words in total**, so one word is 0.8 WER points: this test set cannot resolve
anything under about 5 points, whatever the arm. Utterance bootstrap, 10,000 draws, on the per-clip CSVs.

| system (all `--augment 0.5`) | WER raw | WER +snap | CER +snap |
|---|---|---|---|
| base model | 48.8 % | 47.9 % | 20.3 % |
| A `lora-aug` (LoRA r32, patient audio) | 30.6 % | 25.6 % | 7.5 % |
| B `lora-aug-synth` (A + 300 generic-voice sentences) | 24.8 % | 19.8 % | 8.2 % |
| `lora-r16` | 21.5 % | 17.4 % | 5.3 % |
| `lora-dora-r32` (the README's system) | 15.7 % | 13.2 % | 4.1 % |

- **B vs A:** −5.8 points (+snap), 95 % interval [−14.9, +3.1], P(B ≥ A) = 0.12 → **no evidence either way**, as
  predicted. CER went up slightly (7.5 → 8.2): the generic voices taught words, not his sounds.
- **DoRA vs A:** −12.4 [−21.7, −3.8]: the recipe matters more than any synthetic data so far. The README's
  13.2 / 10.7 figures come from the whisper.cpp + prompt path (`ivg-demo --compare`), not this HF path, so the
  two tables are not the same pipeline; within this one, the DoRA adapter is the baseline arm C must beat.
- **Consequence for arm C:** to be readable at all, C has to move WER by more than ~10 points on 121 words, or
  the test set has to grow. The honest order is: train C on the DoRA recipe (`--dora`, the README's system),
  score it; if the interval straddles zero, the answer is "this test set cannot tell", not "the clone did
  nothing", and the next recordings should add held-out sentences before any further augmentation claims.

## 2026-10-09 — a target to aim at: SAPC2 Track 2 (streaming dysarthric ASR), NeurIPS 2026 workshop

https://xiuwenz2.github.io/SAPC2-website/ and the kit https://github.com/xiuwenz2/SAPC-template

**What it is.** Speech Accessibility Project Challenge 2 (Illinois, Google, Amazon, Apple, Microsoft). Track 1:
unconstrained ASR ranked by accuracy. Track 2: streaming ASR on a Pareto chart of accuracy vs latency,
**evaluated on CPU only**, 100 ms chunks (1600 samples at 16 kHz, float32 mono), two passes (Pass 1 batch for
accuracy, Pass 2 real-time with partial callbacks for latency). Primary metric CER, secondary WER, each utterance
scored against two references (with/without disfluencies, lower kept), clipped at 100 %. Latency = mean of
TTFT-stable (speech onset → first word final) and TTLT (audio end → final output), P50. Prize US$10k split over
the Pareto frontier on the sequestered Test2. Baseline in the kit: a streaming Zipformer-transducer (icefall).

**The Track 2 reject policy, which is the part that matches how we already work.** A submission is excluded if
(1) its stable sentence-prefix match rate on Test1 is under 1/3 (partials that keep changing), (2) on more than
5 % of utterances the first word is final *before speech begins* ("a word cannot be recognized before it is
spoken, so this indicates the first word was guessed rather than recognized"), or (3) Pass 1 and Pass 2 give
different final transcripts after normalisation. Checker: `utils/track2_reject_check.py`. Our guard (never act
on an uncertain command, ask instead) and the sound-based correction (deterministic) are the same discipline:
emit nothing you will retract.

**Interface to implement** (`model.py`, class named exactly `Model`): `__init__`, `set_partial_callback(cb)`,
`reset()`, `accept_chunk(audio_chunk: np.ndarray) -> str` (returns the current partial, calls the callback),
`input_finished() -> str`. All calls from one decoder thread. Zip: `model.py`, `setup.sh`, `requirements.txt`,
weights. Runtime image PyTorch 2.5 CPU; 15,000 s per submission. Local run:
`python3 local_decode.py --submission-dir ... --manifest-csv ... --out-csv ... --out-partial-json ...`.

**Dates.** Competition deadline **24 October 2026**; paper 31 October; workshop 11–12 December, Sydney.
Data access needs a Data Transfer and Use Agreement plus a one-page proposal to
speechaccessibility@beckman.illinois.edu, "typically 2–4 weeks". Sent today, data arrives around the deadline:
**a scored SAPC2 entry this round is unlikely; the DUA is worth sending anyway** (the corpus is the largest
etiology-balanced dysarthric set there is, and the next edition will use it too).

**What "trying to win" means for us without the data, honestly.**
- Our architecture is on the wrong side for Track 2 as it stands: Whisper is not a streaming model. whisper.cpp's
  stream mode re-decodes a sliding window, so partials change, which is exactly rule (1). The kit's baseline
  (Zipformer-transducer) is streaming-native. Track 1 (unconstrained, accuracy only) fits Whisper + personal
  adapter; Track 2 fits the product (phone satellite, CPU, deployable). Decide which one we are aiming at before
  building; the product says Track 2.
- What transfers regardless: the CPU-only constraint (our GGML route), the two-reference CER scoring (add it to
  `ivg-eval`), the reject checker as a test on Christian's clips (run `track2_reject_check.py` on our own
  partials: if whisper.cpp stream fails rule (1) on his audio, we know before any data arrives), and the
  per-speaker adapter as the thing nobody else in the challenge will have (the challenge is speaker-independent;
  our angle is a paper on what personalisation adds on top of the best speaker-independent system).
- Steps that need no SAP data: (a) wrap the current pipeline in the `Model` interface; (b) run the kit's
  `local_decode.py` and the reject checker on the 39 test clips; (c) measure TTFT-stable and TTLT on CPU; (d) if
  rule (1) fails, test a stable-prefix policy (emit a word only when it has survived N chunks), which is also what
  the home assistant needs. Each is a day or less on the Mac.
- Deadline realism: between 9 and 24 October the Munich week is in the middle. (a)–(c) can run in the background;
  (d) and any training wait.

**Action for Ale:** send the DUA request and one-page proposal now (draft in `docs/sapc2_proposal_draft.md`).

**PER added 2026-10-09** (`ivg-per`, German IPA via gruut, 657 reference phones on the 39 clips; `results/.../per.json`):
base 23.6 % (+snap 24.2); A `lora-aug` 12.5 / 11.0; B `lora-aug-synth` 10.2 / 9.9; `lora-r16` 6.8 / 6.4;
`lora-dora-r32` 6.1 / **5.8**. B vs A on PER (+snap): −1.1 points [−5.3, +3.5], P(B ≥ A) = 0.33 → no evidence, same
as WER. PER has five times the resolution of WER here (657 phones vs 121 words), so it is the primary metric for
arm C; the reading rule is unchanged (interval of the paired difference excludes zero). Note the correction step
lowers WER but not PER for the DoRA system (6.1 → 5.8): it fixes spellings, which is what a word-level metric
rewards and a phone-level one barely sees.
