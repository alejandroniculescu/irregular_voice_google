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
  Neuphonic's own licence (NeuTTS Open License 1.0); Ale has reviewed and approved it (noted 2026-10-09).
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

## 2026-10-09 — E3: the best local model for each expert slot (written before any number)

The product is a router over experts, not a mixture of experts: acoustic (Whisper + adapter), lexical (snap:
Kölner Phonetik + spelling distance), command (`home.py` with the guard), dialogue (booking questions with
grammars), open-text fallback (`llmfix`: a local Ollama model whose word swaps pass a sound gate). Routing is
deterministic and stays so; only the fallback slot has a model to choose. Nothing leaves the device: candidates
are local models only. On ahms today (Ollama): `qwen3.5:4b`, `qwen3.5:9b`, `gemma4:26b` (plus two hackathon
fine-tunes that are not ours and are excluded).

**E3a, the fallback as a sound-gated fixer, measurable now.** `ivg-eval --snap --llm <model>` on the 39 test clips
with the DoRA adapter: PER (primary), WER, CER after snap alone vs after snap + each model's gated fix. Reading per
model: helps / hurts / no evidence by the paired PER bootstrap against snap alone; a model that *hurts* PER is out
regardless of WER (it is inventing sound-alikes). Prediction: all three "no evidence" on PER (the gate allows one
sound-alike swap per word, so the ceiling is the handful of real-word confusions in 121 words); gemma4:26b the
slowest by far and no better. Also recorded: seconds per utterance per model, since the fallback runs on a phone
later and 26 B does not.

**E3b, the fallback as an intent reader, needs a labelled set first.** Utterances the command and dialogue experts
did not claim: the model may only produce a clarifying question or a confirmed intent from a closed list, never an
action. Metric: on a set of unclaimed utterances with hand-labelled intended meaning (to be built from the next
recordings and the 15-command replay's near misses), rate of correct intent, rate of wrong action (must be 0),
asking rate. Not runnable until that set exists.

**Per-expert error attribution, from the same CSVs.** For every test utterance: phones wrong after the acoustic
expert (raw PER), after the lexical expert (snap), after the fallback (llm): "where was each error born and which
expert removed it". One table per system; the slide that says which part fails. Added to `ivg-per` as
`--stages raw snap llm`.

Order: arm C finishes (GPU) → E3a (Ollama on the same GPU, so not concurrent) → attribution table. All on ahms,
none needs Ale.

**Per-expert attribution, first numbers (2026-10-09, `ivg-per --stages`, 657 phones).** DoRA r32: raw 40 phones
wrong (6.1 %); after snap 38 (5.8 %): snap **fixed 8 and broke 6**, 3 utterances better, 5 worse. Plain LoRA r32:
raw 82 (12.5 %); after snap 72 (11.0 %). Reading: the lexical expert earns its place on a weaker acoustic expert
and is nearly net-zero on the best one; what it breaks is a real word swapped for a sound-alike that was not
said. That is the E3a question in miniature, and the sound gate on the LLM fallback has the same risk, measured
the same way.

**Error profile by phone class and the oracle bound (2026-10-09, `ivg-per --profile / --oracle`).**
DoRA r32 (+snap), sub+del rate per class: vowels **8.3 %** (18 of 229 substituted), fricatives 4.3 %, plosives
3.4 %, liquids 1.9 %, nasals 0 %. Base model: vowels 23.6 %, liquids 28.8 %, plosives 18.9 %, nasals 17.5 %,
fricatives 12.9 %. So the adapter learned his consonants almost completely and his vowels least: what remains is a
vowel-quality problem, 18 substitutions, not deletions. (Caveat: classes from gruut's IPA, no stress or
syllable position; "other" is 5 phones.)
Oracle over the four adapters, best system per utterance: **PER 1.7 % against 5.8 % for the best single**; the
DoRA system is the pick on 31 of 39 utterances, r16 on 6, synth on 2. Pairwise shared errors: DoRA ∩ r16 only 16
of DoRA's 38, i.e. the two recipes miss *different* sounds. That is the Venn diagram: a router that could tell
which adapter to trust per utterance would take two thirds of the remaining errors away; nothing in the pipeline
can do that today (the cue would have to come from the audio or from agreement between the two decodes), and
that is the E2 structure again, from the error side. Added to E2's reading: the oracle bound is the number any
mixture must be compared against.

### E1, arm C scored 2026-10-09 (`results/20261009-154244/`): the cloned voice hurts

Reference clip: the longest 4–14 s train-split recording; 300 sentences synthesised (NeuTTS Nano German + NeuCodec,
both gated on HF, accepted by Ale), DoRA recipe identical to `lora-dora-r32` plus `--extra-manifest`. Training
reached train WER 0.0 % against dev 25.5 %: it memorised.

| system (+snap) | WER | CER | PER |
|---|---|---|---|
| DoRA r32 (baseline) | 13.2 % | 4.1 % | 5.8 % |
| C: DoRA r32 + 300 cloned-voice sentences | 23.1 % | 7.4 % | 8.8 % |

PER difference C − baseline: **+3.0 points [−0.7, +7.5], P(C ≥ baseline) = 0.94** → under the rule, no evidence
on PER (interval touches zero), hurts on WER (+9.9). Prediction held: the clone copies timbre, not articulation.
The profile says where: the clone damaged the **consonants** the real adapter had learned (plosives 3.4 → 6.9 %,
fricatives 4.3 → 7.9 %, liquids 1.9 → 11.5 %, nasals 0 → 5.3 %) while vowels barely moved (8.3 → 9.2 %). A codec
clone of a dysarthric voice sounds like him and articulates like the model; training on it teaches consonants he
does not produce. Oracle of the two: 3.5 % vs 5.8 %, 23 shared errors, so C is not useless as a second opinion,
but it is not a data source. **Voice banking stays a feature (the assistant's reply voice), not training data.**
Closed; the next augmentation claim waits for more held-out words.

### E3a, scored 2026-10-09 — the local LLM fallback hurts; keep it out of the default chain

First run was inert (thinking models returned empty responses; three models identical to four digits; fixed in
be7f34a with `think: false`, probe in `scripts/llmfix_probe.py`). **Rule added: identical outputs across arms are a
bug until shown otherwise.** Rerun on the DoRA r32 adapter, 39 clips, 657 phones (`results/20261009-1609…1610…`):

| fallback model | WER | CER | PER | PER vs snap alone [95 %] | phones fixed / broken |
|---|---|---|---|---|---|
| none (snap only) | 13.2 % | 4.1 % | 5.8 % | — | (snap: 8 / 6) |
| qwen3.5:4b | 15.7 % | 5.1 % | 7.5 % | +1.7 [−0.2, +3.8] | 2 / 13 |
| qwen3.5:9b | 14.1 % | 4.1 % | 6.1 % | +0.3 [−1.0, +1.7] | 4 / 6 |
| gemma4:26b | 14.1 % | 5.1 % | 7.3 % | **+1.5 [+0.2, +3.1]** | 0 / 10 |

Reading: none helps; gemma 26b **hurts** (interval excludes zero), qwen 4b nearly; qwen 9b is the least harmful
and still net negative. The biggest model fixed nothing and broke ten phones: through a sound gate, a stronger
language model is more confident in sound-alike real words the speaker did not say. Prediction ("no evidence")
was too kind. **Decision for the product: the LLM is not a fixer.** If it stays, it stays in the one role where
being wrong is safe: proposing a clarifying question, never rewriting a transcript (E3b, needs a labelled set).
The acoustic side is where the gain is (oracle 5.8 → 1.7 % over adapters).

## 2026-10-09 — R1: a reference-free router over the existing adapters (written before any number)

The oracle says 5.8 → 1.7 % if something picks the right adapter per utterance. Not all of E2 needs new recordings:
a router over the **four adapters we already have** can use cues that exist at inference time. Three routers with
no trained parameters (39 clips cannot fit any), scored from the `+snap` CSVs in `results/20261009-132109/`:
**medoid** (the hypothesis with the smallest total phone edit distance to the others), **ROVER** (word-level vote
after aligning every hypothesis to DoRA's, ties to DoRA), **switch** (DoRA unless two other systems agree with each
other, word for word, against it). Code in `sonic/route.py`. A fourth, **confidence** (pick the decode with the
highest mean token log-prob), needs a rerun on ahms that logs log-probs; next.
Reading: each router against DoRA r32 +snap by the paired PER bootstrap (WER, CER and the vowel error rate reported alongside; vowels are where DoRA is weakest, 8.3 %); and the share of the oracle gap
(5.8 → 1.7) it closes. Prediction: medoid and ROVER *hurt* or no evidence (three of the four systems are much
weaker than DoRA, 6.4–11.0 %, so consensus pulls toward their shared errors); switch no evidence, closing under a
fifth of the gap. If so, agreement is not the cue and confidence is the one to test.
Cost note for Track 2: N adapters = N decodes on CPU; a router only earns its place if it pays for that.

### R1 scored 2026-10-09 (`results/20261009-132109/route*.json`): medoid helps, the prediction was wrong
Pivot DoRA r32 +snap: WER 13.2 %, CER 4.1 %, PER 5.8 %, vowel error 8.3 %. Routers over pivot + r16 + aug-synth + aug:

| router | WER | CER | PER | vowel | PER diff [95 %] | P(diff ≥ 0) | oracle gap closed | clips changed |
|---|---|---|---|---|---|---|---|---|
| medoid | 9.1 % | 2.6 % | **3.3 %** | 5.2 % | −2.4 [−5.6, 0.0] | 0.034 | 59 % | 9 |
| rover | 10.7 % | 3.0 % | 4.3 % | 6.1 % | −1.5 [−4.7, +1.0] | 0.146 | 37 % | 7 |
| switch | 11.6 % | 3.2 % | 4.6 % | 6.6 % | −1.2 [−4.1, +0.9] | 0.184 | 30 % | 7 |

Medoid is stable to the system set: PER 3.3 % with 3 systems (pivot + r16 + aug-synth), 4, or 5 (+ neutts, where
ROVER also reaches 3.3 %). Reading: medoid **helps** on PER by the rule (P = 0.03, interval touching 0, 39 clips,
so a weak "helps"), and it moves WER, CER and the vowel rate the same way; ROVER and switch no evidence. The
prediction (consensus of weaker systems pulls toward shared errors) was wrong: the weaker adapters err in
*different* places, so the one closest to all others is usually the right one. The vowel rate falls most
(8.3 → 5.2 %), i.e. the router fixes the class the single adapter is worst at.
Caveats: same 39 clips the oracle was read on, no held-out set; medoid has no fitted parameters, so this is not
overfitting a router, but the system set was chosen knowing these clips. Cost for Track 2: 3 decodes per
utterance on CPU; the 3-system medoid is the cheapest that keeps the full gain.

## 2026-10-09 — S1: the router on CPU whisper.cpp, and what the encoder window costs (written before any number)

Track 2 is scored on CPU, and on CPU the encoder is the whole cost: one padded 30 s encode of large-v3-turbo q5_0 is
8.7 s via `whisper-cli` on the M3 (4 threads), the decode 0.06 s. Shrinking `audio_ctx` to the audio cut encode 11×
but made the decoder loop (114 → 3845 decode steps on 8 clips). `sonic/engine.py` runs whisper.cpp in-process
(model loaded once, greedy, token cap 8/s of audio) and returns mean token log-prob; `sonic/bench.py` scores the
three medoid experts (DoRA r32, r16, aug-synth, merged, q5_0) at encoder window full (0), auto (audio + 1 s) and
512 (≈10 s), raw (no snap), then medoid and **confidence** (highest mean log-prob) routers over them.
Reading: per window, WER/CER/PER/vowel against the same expert at full window; a window is usable if PER rises by
under 1 point. Router: medoid vs confidence vs DoRA alone at the chosen window. Prediction: whisper.cpp at full
window within a point of the HF numbers (DoRA raw PER 6.1 %); auto window *hurts* (fine-tuned on padded audio);
512 in between; confidence no better than DoRA alone (Whisper's log-probs are overconfident on fluent wrong words),
medoid holds most of its gain from R1.

### S1 scored 2026-10-09 (`results/sonic/20261009-174739/`): confidence routing wins; the short window breaks
39 clips, raw (no snap), whisper.cpp q5_0 in-process, CPU, 4 threads on the M3.

| system | WER | CER | PER | vowel | sec/clip |
|---|---|---|---|---|---|
| DoRA @ full window | 19.8 % | 5.6 % | 6.1 % | 4.5 % | 3.29 |
| r16 @ full | 22.3 % | 5.7 % | 5.9 % | 3.1 % | 3.62 |
| aug-synth @ full | 26.4 % | 8.5 % | 9.4 % | 5.4 % | 3.42 |
| DoRA @ 512 (≈10 s) | 24.0 % | 7.2 % | 7.6 % | 4.9 % | 0.86 |
| DoRA @ auto (audio + 1 s) | 162.8 % | 111.6 % | 106.6 % | 44.2 % | 0.38 |
| medoid of 3 @ full | 16.5 % | 4.2 % | 4.5 % | 3.1 % | 10.3 |
| **confidence of 3 @ full** | **12.4 %** | **2.8 %** | **3.0 %** | **1.8 %** | 10.3 |
| confidence of 3 @ 512 | 21.5 % | 5.7 % | 6.1 % | 3.1 % | 2.67 |

Readings against the predictions. whisper.cpp at full window equals HF (DoRA raw PER 6.1 % both): as predicted.
Auto window **breaks** (the decoder loops or invents; worse than predicted); 512 costs +1.5 PER points
(P = 0.94 worse), over the 1-point rule, so **the final decode stays at the full window**; 512 is for partials
only. Confidence router vs DoRA: **helps**, −3.0 points [−6.6, −0.8], P < 0.001; vs medoid: helps, −1.5 points,
P = 0.025. The prediction (Whisper over-confident, confidence no better than DoRA) was wrong: per-adapter mean
log-prob is a usable cue between adapters of the same base. Medoid on CPU decodes is weaker than on the HF decodes
in R1 (4.5 vs 3.3 %), so R1's ranking does not carry over between decoders. Oracle over the three: 1.2 %.
Cascade (offline, same decodes): DoRA first, escalate to all three only if DoRA's mean log-prob < t. t from −0.02 to
−0.08 keeps PER 3.0 % with 29 → 19 of 39 clips escalated (mean 8.5 → 6.7 s); t = −0.20 gives 4.0 % at 4.6 s.
Chosen t = −0.05, mid-plateau, since t was read on these clips. Cost is the open problem: a full-window encode is
3.3 s per expert on 4 M3 cores; TTLT ≈ 3.3 s (confident) to 10 s (escalated) unless the experts run in parallel.

## 2026-10-09 — C1: how early could a closed-set command expert commit? (cohort uniqueness points; written before any number)

Idea (Ale): a deterministic command expert that commits as soon as the heard prefix leaves one command, instead of
waiting for the utterance to end (Marslen-Wilson's cohort model: a word is recognised at its uniqueness point).
Measured without any model: every canonical home command (`home.phrase`: device + room + action over
`home_de.json` and `ACTIONS`, about 110) and every short command in `commands_de.txt`, in gruut IPA; uniqueness
point = phones until no other command shares the prefix; saving = phones after it ÷ his speaking rate (phones per
second of voiced time on the 39 test clips, energy onset to offset). Second variant with vowels as wildcards,
since vowels are his weakest class (8.3 % vs ≤ 4.3 % for every consonant class): a commit that needs a vowel to
decide is a commit on his least reliable sound.
Reading: median and spread of the saving in ms per set. Worth building for the demo if the median saving on the
home commands is ≥ 500 ms with vowels as wildcards. Prediction: no: the deciding word is the action, which comes
last, and an/aus differ in the vowel; median saving under 200 ms, near 0 with vowel wildcards. The short commands
may do better (single words, unique early). If so, the lever is the phrasing (action first: "Aus, Licht im Flur"),
not the model.

### C1 scored 2026-10-09 (`sonic/cohort.py`, `results/sonic/cohort.json`): the phrasing is the lever, not the model
His rate on the test clips: 10.0 phones/s (100 ms per phone, energy onset to offset). Uniqueness-point savings:

| command set / order | median saving | p25–p75 | ≥ 500 ms |
|---|---|---|---|
| home, device room action (today: "Licht im Flur an") | 100 ms | 0–500 | 33 % |
| home, room device action | 100 ms | 0–500 | 33 % |
| home, **device action room** ("Licht an im Flur") | **550 ms** | 275–825 | **62 %** |
| home, action device room | 550 ms | 275–825 | 62 % |
| short commands (`commands_de.txt`, 24) | 350 ms | 200–624 | 29 % |

Vowels as wildcards change nothing (today 1 phone, room last 6 phones median): the deciding phones are consonants,
his reliable class. (The first run printed "0 %" at ≥ 500 ms for today's order: the rate is just over 10/s, so
5 phones came out as 499 ms. 33 % is right.)
Reading: the prediction held for today's order (median 100 ms: the action comes last and decides). Every slot
carries information, so a command can only become unique inside its last word; the order decides which word that
is. With the room last (long, consonant-distinct words), the median command could fire 550 ms before he finishes:
passes the 500 ms bar. `home.parse` already reads slots in any order, so "Licht an im Flur" needs no parser change,
only a suggested phrasing. Next: a prefix-commit command expert in `sonic/` fed by the streaming partials, judged
on the 15-command replay (0 wrong actions stays the hard constraint) and on ms saved in practice, which will be
less than this bound (the partials lag the audio by one decode).

## 2026-10-09 — S2: the kit run, and the latency fixes (first run scored; fixes written before their numbers)
First `local_decode.py` run on the 39 clips (`results/sonic/kit/`, M3, 4 threads, partials every 0.5 s at 512,
final = confidence cascade at full window, t = −0.05): WER 14.0 %, CER 3.1 %, **PER 3.3 %**, vowel 1.8 %; reject
rule 1 (early first word, energy onset) 0/39, rule 2 (Pass 1 = Pass 2 finals) 39/39: **both pass**. Latency fails
any sensible bar: TTFT-stable median 5.3 s (p90 15.7), TTLT median 13.3 s. Cause: backpressure, a 0.86 s partial
every 0.5 s of audio, so chunks queue; then a 3.3–10 s final. Parallel experts: the bindings release the GIL
(2 threads × 3 experts: 14.0 → 8.7 s) but 4 M3 performance cores are the ceiling; it pays only on a many-core box.
Fixes: (a) shed partial decodes while the decoder lags the audio by > 0.2 s; (b) a final window of 768 or 1024 if
one stays within the 1-point PER rule. Prediction: (a) brings TTFT-stable under 2 s median and TTLT to the final's
own cost (≈ 3.3 s confident, ≈ 10 s escalated, median ≈ 5 s); (b) 1024 within a point, 768 not.
S2 (b) scored (`results/sonic/20261009-181827/`): DoRA alone is window-insensitive from 768 up (768 +0.3, 1024
−0.1 PER points vs full); the confidence router is not: 768 **+2.2** [+0.6, +4.3] (fails the 1-point rule),
1024 +0.6 [+0.1, +1.1] (PER 3.6 %, WER 15.7 %, CER 3.3 %, vowel 1.8 %; passes). Per-expert cost 3.3 → 2.0 s.
Reading: the log-probs that do the routing are more window-sensitive than the argmax text. Final window → 1024,
threshold left at −0.05 (calibrated at full; not refit on the same clips). Prediction (b) held for 1024; for 768
it held for DoRA alone and failed for the router.
S2 (a) scored, second kit run (`results/sonic/kit2/`: shedding on, final window 1024): WER 16.5 %, CER 3.7 %,
PER 4.0 %, vowel 2.2 %; TTFT-stable median **3.35 s** (p90 8.7), TTLT median **6.0 s** (p90 7.1); rules 1 and 2
pass. Reading: shedding and the shorter final more than halved both latencies, but missed the prediction (TTFT <
2 s): a partial decode still costs ~0.9 s and the gate waits for 200 ms of speech plus the first decode.
PER 4.0 vs 3.6 % in the bench at 1024: the cascade threshold was calibrated at the full window, and log-probs
shift with the window, so fewer clips escalate. Next (docs/research_2026-10-09.md): soup, quantization sweep,
speculative final, short-window training.

## 2026-10-09 — S3: soup, quantization on x86, short-window adapter (ahms i9-14900K, 4 threads, CPU)
Predictions (before the run, from docs/research_2026-10-09.md): soup within reach of the router (MAS-LoRA); q4_0
faster than q5_0 on x86 (whisper.cpp #3752); the short-window adapter works at a fitted window (ACFT-like).

| model @ window | WER | CER | PER | vowel | sec/clip |
|---|---|---|---|---|---|
| DoRA q4_0 @ full | 23.1 % | 6.3 % | 6.7 % | 4.9 % | 5.80 |
| DoRA q5_0 @ full | 20.7 % | 5.9 % | 6.5 % | 4.5 % | 10.32 |
| DoRA q8_0 @ full | 18.2 % | 5.3 % | 5.8 % | 4.5 % | 7.36 |
| DoRA q4_0 @ 1024 | 25.6 % | 6.3 % | 6.7 % | 4.5 % | 3.59 |
| **soup (DoRA+r16+aug-synth, uniform) q5_0 @ full** | **13.2 %** | **3.3 %** | **3.6 %** | 3.1 % | 10.33 |
| soup q5_0 @ 1024 | 14.0 % | 4.0 % | 4.3 % | 3.6 % | 6.69 |
| short-window DoRA q5_0 @ fit (floor 128) | 76.9 % | 47.2 % | 47.2 % | 25.0 % | 1.20 |
| short-window DoRA q5_0 @ 512 / 1024 / full | 18.2 / 16.5 / 17.4 % | | 8.0 / 5.8 / 5.9 % | | 3.24 / 7.10 / 11.20 |

Readings. **Soup helps**: one pass, PER 3.6 % vs DoRA 6.5 % on the same box, against 3.0 % for the 3-pass
confidence router (S1, M3): most of the mixture's gain at a third of the cost; prediction held. Quantization:
on x86 q5_0 is the slowest format (q4_0 1.8×, q8_0 1.4× faster) and q8_0 also the most accurate; prediction held
for q4_0, q8_0 a surprise. The i9 at 4 threads is ~3× slower than the M3 for the same model (10.3 vs 3.3 s at
q5_0): if the evaluation box is x86, every latency above triples. (Corpus downloads ran on one core alongside.)
Short-window adapter: **failed** at a fitted window in whisper.cpp although HF dev WER was fine during training;
suspects: HF `generate` padding the cropped dev features back to 3000 (so dev never tested short windows), or a
crop mismatch with whisper.cpp. Parked; not on the critical path.
Next: soup at q8_0 and q4_0; the Track 2 model becomes soup-first (one pass), with the experts as the escalation.

### S3 soup quantization scored (`ahms:results/sonic/20261009-223909/`, i9, 4 threads)
**Correction (all S3 PER and vowel numbers above are wrong):** ahms `.venv-sonic` had gruut without
`gruut_lang_de`, so gruut returned no phonemes for any German word and `per.gruut_g2p` silently fell back to letters:
S3's "PER" was a letter error rate. WER/CER were right. Fixed: the pack is installed and `gruut_g2p` now raises when
a language pack is missing. Rescored with German IPA (same CSVs):

| model @ window | WER | CER | PER | vowel | sec/clip |
|---|---|---|---|---|---|
| DoRA q4_0 / q5_0 / q8_0 @ full | 23.1 / 20.7 / 18.2 % | 6.3 / 5.9 / 5.3 % | 8.8 / 8.1 / 7.2 % | 8.3 / 7.0 / 6.6 % | 5.8 / 10.3 / 7.4 |
| DoRA q4_0 / q5_0 / q8_0 @ 1024 | 25.6 / 21.5 / 19.0 % | | 9.9 / 8.5 / 8.1 % | 8.7 / 8.7 / 8.7 % | 3.6 / 6.7 / 4.7 |
| soup q5_0 @ full / 1024 | 13.2 / 14.0 % | 3.3 / 4.0 % | 4.6 / 5.2 % | 6.1 / 6.1 % | 10.3 / 6.7 |
| soup q4_0 @ full / 1024 | 15.7 / 18.2 % | 4.3 / 4.8 % | 5.5 / 6.8 % | 7.0 / 7.4 % | 6.8 / 5.3 |
| **soup q8_0 @ full** | **13.2 %** | **3.5 %** | **4.1 %** | **4.8 %** | 8.8 |
| soup q8_0 @ 1024 | 16.5 % | 4.6 % | 5.9 % | 6.6 % | 5.0 |
| short-window DoRA q5_0 @ fit / 512 / 1024 / full | 76.9 / 18.2 / 16.5 / 17.4 % | | 49.9 / 8.8 / 6.8 / 6.8 % | | 1.2 / 3.2 / 7.1 / 11.2 |

Readings hold, numbers move: the soup roughly halves DoRA's PER (4.1–4.6 vs 7.2–8.1 %); q8_0 is the most accurate
format and faster than q5_0 on x86; q4_0 costs ~1.5 points; the soup loses ~1.8 points at 1024. Track 2 final: soup
q8_0 at the full window. Note for comparisons: whisper.cpp numbers here are **without snap**; the HF reference
(DoRA PER 5.8 %) is after snap. DoRA q5_0 @ full is 7.6 % on the M3 and 8.1 % on the i9 (same file, 4/39 clips
decode differently): no platform bug.

## 2026-10-09 — S4: the cascade threshold with the soup first (written before any number)
`track2/model.py` now decodes the soup first and escalates to the three experts (DoRA, r16, aug-synth; q5_0) when
the soup's mean token log-prob is below t; the most confident of all four wins. t = −0.20 was inherited from the
DoRA-first cascade and never fit for the soup. Offline, from per-clip CSVs at the full window (soup q8_0 from ahms,
experts q5_0 from S1 on the M3; log-probs of the same model and format should agree across machines): PER, vowel
error and share of clips escalated for t from −0.40 to 0. Second arm, added after the gruut correction and before
its numbers: the same sweep with `snap.for_speaker` (train transcripts, commands, lexicons; never test) on the
final; predicted −0.5 to −1.5 PER points, since snap was in the HF reference and costs milliseconds.
Reading: ship the t in the middle of a plateau that beats the soup alone by ≥ 0.5 PER points with ≤ 50 % of clips
escalated. If none does, the final is the soup alone, and escalation stays only as a visible demo path with a t that
fires rarely (routing shown, not sold as a gain). t is read on the test clips, so any gain is optimistic.
Prediction: no t passes. The soup already holds most of the mixture's gain (3.7 vs the router's 3.0 %), and
averaged models tend to be more confident, so argmax over log-probs will mostly keep the soup; gain < 0.5 points.

### Correction: every sonic PER and vowel number before this point was a letter error rate (gruut had no German)
The same missing `gruut_lang_de` hit the Mac's `.venv-sonic` (S1, S2, kit, kit2, C1). Rescored from the saved CSVs
with German IPA (`.venv-sonic`, pack installed; WER/CER unchanged):

| run | WER | CER | PER (was) | vowel (was) |
|---|---|---|---|---|
| S1 DoRA / r16 / aug-synth @ full (M3, q5_0) | 19.8 / 22.3 / 26.4 % | 5.6 / 5.7 / 8.5 % | 7.6 / 7.6 / 10.8 % | 7.0 / 7.4 / 9.2 % |
| S1 confidence router @ full | 12.4 % | 2.8 % | **3.8 %** (3.0) | 3.9 % |
| S1 medoid @ full | 16.5 % | 4.2 % | 6.1 % | 5.2 % |
| S2 confidence router @ 1024 / 768 | 15.7 / 16.5 % | 3.3 / 4.7 % | 4.9 / 6.2 % (3.6 / —) | 4.8 / 5.2 % |
| kit (full window) | 14.0 % | 3.1 % | **4.0 %** (3.3) | 3.9 % (1.8) |
| kit2 (shedding, 1024) | 16.5 % | 3.7 % | **5.3 %** (4.0) | 4.8 % (2.2) |

The S1/S2 readings survive in direction (router ≫ any single expert; 768 fails, 1024 costs ~1.1 points). The S1
threshold t = −0.05 was fit on letter rates; S4 refits it for the soup. C1 rerun (`sonic/cohort.py` now computes
the four slot orders itself): his rate 9.1 phones/s (was 10.0); today's order saves a median 221 ms (0 % ≥ 500 ms);
**device action room** 552 ms (62 % ≥ 500 ms; 497 ms / 50 % with vowel wildcards); short commands 331 ms. The C1
reading holds: put the room last.

### S4 scored (`scratchpad` sweep over `results/sonic/ahms_a1/soup_q8@0.csv` and S1's expert CSVs)
| final | WER | CER | PER | vowel | clips escalated |
|---|---|---|---|---|---|
| soup q8_0 alone | 13.2 % | 3.5 % | 4.1 % | 4.8 % | 0 |
| cascade t = −0.20 | 12.4 % | 2.7 % | 3.7 % | 4.4 % | 2/39 |
| cascade t = −0.05 | 12.4 % | 2.7 % | 3.8 % | 4.4 % | 12/39 |
| soup alone + snap | 9.1 % | 2.7 % | 3.7 % | 4.8 % | 0 |
| **cascade t = −0.20 + snap** | **8.3 %** | | **3.2 %** | 4.4 % | 2/39 |
| cascade t = −0.05 + snap | 9.1 % | | 2.9 % | 4.8 % | 12/39 |

Soup log-probs: median −0.027 (experts −0.08 to −0.24), so the soup is the confident one, as predicted. Arm 1: no t
beats the soup by ≥ 0.5 PER points (best −0.4 at t ≤ −0.15): **prediction held**; escalation stays as the demo's
routing path at t = −0.20, firing on ~5 % of clips. Arm 2: snap −0.4 PER points on the soup (just under the
predicted −0.5 to −1.5) but WER 13.2 → 9.1 %; vowels untouched (snap fixes spelling of consonant clusters). With
snap, t = −0.05 reaches 2.9 % PER at six times the escalation; one-clip noise at 657 phones, and WER is worse.
Shipped: soup q8_0 first, t = −0.20, snap on the final (`SONIC_SNAP`).

## 2026-10-09 — S5: kit run 3, soup-first + snap (written before the run)
`local_decode.py` on the 39 clips, M3, 4 threads: partials from the soup q8_0 (auto window, shedding), final =
cascade at the full window, t = −0.20, snap. Reading: accuracy should reproduce S4's offline row within a clip's
noise (it is the same decode, now in the streaming harness); latency is the point. Prediction: WER ≈ 8–9 %, PER ≈
3.2 %; rules 1 and 2 pass; TTLT median ≈ the soup's full-window cost on the M3 (≈ 3 s; the speculative final helps
only on clips with ≥ 0.3 s of trailing silence after trim); TTFT-stable median ≈ 3 s (partials as costly as kit2's).
If TTLT > 4 s median, the next lever is the final window, not the router.

### S5 scored, accuracy only (`results/sonic/kit3/`, M3): latency void, machine contended
WER **9.1 %**, CER 2.4 %, PER **3.5 %**, vowel 4.8 %; rules 1 and 2 pass. Accuracy prediction held (S4 offline:
8.3 % / 3.2 %, one clip apart). Latency read TTFT-stable 13.6 s, TTLT 11.5 s median, but the Mac's load average was
7.8–9.7 during the run (other processes): the same DoRA q5_0 full-window decode took 7 s instead of S1's 3.3 s, soup
q8_0 9 s. TTFT > TTLT means no partial was emitted before the final: under that load every partial was shed. Not
read. Rerun on ahms pinned to 4 cores (`taskset`), idle box (S5a).

### S5a (`ahms:results/sonic/kit3a/`, i9 pinned to 4 P-cores): rule 2 failed; latency void again
WER 9.1 %, CER 2.4 %, PER 3.5 %, vowel 4.8 % (same as the M3). **Rule 2: 1/39 mismatch** after the kit's
normalization (2/39 raw, one word each): Pass 1 and Pass 2 finals differed. Another user's 13-core job ran
alongside (load ≈ 20): TTFT 16.2 s, TTLT 14.6 s median, not read.
Cause, isolated on test011: the same soup decode of the same samples gave different log-probs run to run
(−0.0800 … −0.0865 at 1 thread), sometimes a different word; zero-padding the input to 30 s ourselves made it
repeat exactly (5/5) and equal to the modal unpadded value, as if whisper.cpp read uninitialized memory past short
input on x86 (never seen on the M3). Fix: `engine.Expert.decode` pads every input to 30 s (window and token cap still
from the real length). Check: kit run 3b on ahms under the same load must give rule 2 = 39/39.
Kit run 3b (`ahms:results/sonic/kit3b/`, padded, same load): **rule 2 39/39**; WER 9.9 %, CER 2.8 %, PER 4.3 %,
vowel 6.1 %. The only finals that changed against 3a are the two unstable clips (test011, test022), so 3a's 3.5 %
was a draw on them; 3b is the reproducible number. Padding is neutral for the soup itself (39 clips × 2 runs, i9):
raw 1/39 clips differ between runs (PER 4.1 / 4.3 %), +1 s pad 1/39 (4.3 / 4.1 %), **30 s pad 0/39 (4.1 / 4.1 %)**,
equal to S3's soup number. Current honest system: soup q8_0 + cascade t = −0.20 + snap = WER 9.9 %, PER 4.3 %,
vowel 6.1 %. Latency still unmeasured on an idle box (both machines were loaded).

### S5b (`results/sonic/kit3c/`, M3, padded engine): latency measured, and it is structural
WER 9.1 %, CER 2.4 %, PER 3.5 %, vowel 4.8 % (the M3 decodes the two x86-unstable clips the lucky way, stably);
rules 1 and 2 pass. TTFT-stable median **12.7 s** (p90 25.1), TTLT **11.2 s** (p90 22.0); 1-min load median 5.6
(min 2.4, our run ≈ 4). Prediction (TTLT ≈ 3 s, TTFT ≈ 3 s) **failed by 4×**, and not from load alone: TTFT > TTLT
on most clips means no partial settles before the final; the full-window soup decode (≈ 4.5 s on the M3) and the
partials (≈ 2 s per re-decode of the buffer) cannot keep up with 0.5 s steps, and the speculative final blocks
`accept_chunk`. Against the public Track 2 board (weekly report 2026-09-30: best 8.83 % CER at 1.29 s total,
takagi 13.66 % at 0.55 s; ranking latency = mean of TTFT-stable and TTLT; see docs/research_2026-10-10.md) a
large-Whisper soup is ~10× too slow on CPU. The architecture, not the knobs, is the lever.

## 2026-10-10 — Direction (Ale): Christian first, live phone calls; Track 2 for the game
SAP data access has no date, so the paper is built on Christian's data; the use case is **speech-to-text for his
live phone calls**, which needs ~1 s from speech to text on modest hardware, the same thing Track 2 rewards (board,
weekly report 2026-09-30: 8.83 % CER at 1.29 s; S5b: our Whisper soup 12.7 s + 11.2 s). Track 2 is played with the
public SAP-tuned English Parakeet (`dys-asr/parakeet-rnnt-0.6b-sapc12-syn`); for Christian the candidate is
NVIDIA `parakeet-tdt-0.6b-v3` (25 European languages incl. German, CC BY 4.0, 0.6B FastConformer-TDT), so one
streaming wrapper serves both.

## 2026-10-10 — P1: Parakeet v3, unadapted, on his 39 test clips (written before any number)
`nvidia/parakeet-tdt-0.6b-v3` via transformers 5.15.1, greedy, CPU, 4 threads (ahms i9 pinned to 4 P-cores and the
M3), offline per clip: CER, WER, PER, vowel error, sec/clip; with and without snap. Compared with the unadapted
Whisper base (`whisper-large-v3-turbo-german`: WER 48.8 %, CER 20.3 % +snap) and the soup (CER 3.5 %, PER 4.1 %,
~4.5 s/clip M3).
Reading: Parakeet is worth adapting to him if unadapted CER ≤ 1.5 × the Whisper base's (≤ 30 %) **and** it decodes a
clip in ≤ 1 s on 4 cores. Prediction: CER 20–35 % (a general model on his speech, as Whisper base) and 0.3–0.8 s per
clip, i.e. 5–10× faster than the soup; vowels its weakest class, as for every model so far.

### P1 scored (`ahms:results/sonic/20261010-094701/`, i9 pinned to 4 P-cores, load ≈ 2.5)
| system | WER | CER | PER | vowel | sec/clip |
|---|---|---|---|---|---|
| Parakeet v3 unadapted | 93.4 % | 48.8 % | 53.9 % | 53.3 % | **0.15** (max 0.26) |
| Parakeet v3 unadapted + snap | 91.7 % | 48.6 % | 53.9 % | 53.7 % | |
| Whisper base unadapted + snap (table above) | 47.9 % | 20.3 % | | | |
| soup q8_0 (S3, same box) | 13.2 % | 3.5 % | 4.1 % | 4.8 % | 8.8 |

Checks before reading: hypotheses are German (115 of 132 words known German, 3 English-only, 1 clip in Cyrillic,
1 empty; 132 words vs 121 reference), and a clean synthetic German sentence (macOS voice "Anna", 7.2 s) came out
word-perfect in 0.38 s, so the pipeline is sound. Reading: **fails the rule** (CER 48.8 % > 30 %): prediction wrong
on accuracy (predicted 20–35 %), and beaten on speed (0.15 s vs predicted 0.3–0.8 s; ~60× the soup on this box).
Unadapted, Parakeet hears him far worse than Whisper does. Whether adaptation closes the gap is a separate question
(the SAP Parakeet shows the architecture adapts to dysarthric speech, with ~830 h; we have his train split only).

## 2026-10-10 — P2: Parakeet v3 fine-tuned on his train split (written before training)
P1 failed its rule, but on a premise that broke (speed 60×, not 5–10×), so one cheap test of adaptation, agreed
with Ale. Full fine-tune of `nvidia/parakeet-tdt-0.6b-v3` (transformers `ParakeetForTDT`, TDT loss) on his 278
train clips, AdamW lr 1e-5, bf16 autocast, one RTX 4090; model selection on the 35 dev clips (CER each epoch, best
kept, stop after 5 epochs without improvement); the 39 test clips scored once at the end, CPU, as P1.
Reading: test CER ≤ 7 % (2 × the soup's 3.5 %) → the fast model is the phone-call candidate and goes into the
streaming wrapper as the model that streams; 7–15 % → it streams the partials and the soup (or Whisper DoRA) does
the final, a two-expert story; > 15 % → park Parakeet for German. Prediction: test CER 8–15 %, PER similar, vowels
worst; 278 clips move a general model a long way (Whisper base 20 → DoRA 4 %), but Parakeet starts from 49 %.

### P2 interim: training stopped by ahms crashes (epoch 16 of ≤ 40), not converged
Run 1 died at the epoch-15 save on a `TypeError` inside the stdlib `re` compiler; run 2 (same seed, reproduced run 1
to the digit) ran to epoch 16, then ahms hard-reset at 12:38:44 (log stops, no shutdown, no MCE logged; i9-14900K,
microcode 0x133). Dev CER: 46.5 % (start) → 67.2 → 61.5 → 52.8 → 44.6 → … → 18.7 (ep 14) → 17.0 → **13.7 % (ep 16)**,
still falling ~2–3 points per epoch. Test, epoch-16 checkpoint (`ahms:results/sonic/20261010-125945/`, CPU 4 cores):

| system | WER | CER | PER | vowel | sec/clip |
|---|---|---|---|---|---|
| Parakeet v3 unadapted (P1) | 93.4 % | 48.8 % | 53.9 % | 53.3 % | 0.15 |
| **Parakeet v3, his train split, epoch 16** | 38.8 % | 16.7 % | 18.9 % | 13.5 % | 0.15 |
| + snap | 36.4 % | 16.7 % | 18.7 % | 15.3 % | |
| soup q8_0 (S3) | 13.2 % | 3.5 % | 4.1 % | 4.8 % | 8.8 |

Not read against the rule yet (training was cut off while improving). Run 3 to completion on the other GPU next.

### P2 scored (run 3, ahms GPU 1, CPUs 2-7 after the faulty-core split; best epoch 32 of 37)
Faulty ahms cores: logical CPUs 0, 8, 10 (cereVox's September crash logs); run 1's `TypeError` in `re` and the
12:38 hard reset happened with CPU 0 in use. Run 3 used GPU 1 and CPUs 2-7, ran to its stopping rule (5 epochs
without dev improvement) with no fault. Dev CER: 46.5 % → 11.5 (ep 20) → 9.6 (27) → 8.6 (29) → **7.5 % (ep 32)**.
Test, once (`ahms:results/sonic/20261010-130658/`, CPU 2-5 = 2 P-cores with HT, 4 threads):

| system | WER | CER | PER | vowel | sec/clip |
|---|---|---|---|---|---|
| Parakeet v3 unadapted (P1) | 93.4 % | 48.8 % | 53.9 % | 53.3 % | 0.15 |
| Parakeet v3, epoch 16 (crash-stopped run 2) | 38.8 % | 16.7 % | 18.9 % | 13.5 % | 0.15 |
| **Parakeet v3, his train split (run 3, ep 32)** | **28.9 %** | **8.9 %** | **11.9 %** | **8.7 %** | **0.21** |
| + snap | 27.3 % | 9.5 % | 12.3 % | 10.0 % | |
| soup q8_0 (S3) | 13.2 % | 3.5 % | 4.1 % | 4.8 % | 8.8 |
| Whisper DoRA alone, whisper.cpp q5_0 (S1) | 19.8 % | 5.6 % | 7.6 % | 7.0 % | 3.3 (M3) |

Reading: CER 8.9 % is in the 7–15 % band → Parakeet **streams the partials** and a Whisper expert does the final (a
two-expert system: fast streamer + accurate finaliser). Prediction (CER 8–15 %) held. 278 clips took it from 48.8
to 8.9 % CER, 2.5× the soup's CER at ~1/40 of its cost. Snap hurts it slightly (+0.6 CER): its errors are not the
non-word spellings snap fixes. Not tuned: lr, epochs, augmentation, encoder-only training; dev was still noisy
(±1.5 points epoch to epoch). Timing here is on 2 physical cores (hyperthreaded), slower than P1's 4.

## 2026-10-10 — D1: was the x86 decode jitter the faulty core? (written before the run)
S5a's jitter (and the 30 s padding fix in `engine.py`) was measured pinned to CPUs 0,2,4,6, and CPU 0 is faulty.
Rerun on CPUs 2-5 only: soup q8_0, full window, 4 threads, the 39 test clips, raw input × 3 runs, 30 s pad × 2.
Reading: raw 0/39 clips differ across its 3 runs → the jitter was CPU 0, padding is not needed for determinism (kept
only if neutral, which it was) and S5a's explanation is withdrawn; raw ≥ 1/39 → whisper.cpp itself, padding stays.
Prediction: 0/39 (the M3 never jittered, and the other symptoms on CPU 0 were hardware faults).

## 2026-10-10 — M1: does Parakeet add complementary errors to the mixture? (written before any number)
Idea (Ale): add Parakeet to the MoE, since a mixture gains only where experts err differently, and Parakeet is a
different architecture (FastConformer-TDT) from the three Whisper adapters, which share one base. Offline, per-clip
CSVs on the 39 test clips: Parakeet run 3 (P2), soup q8_0 (S3), DoRA / r16 / aug-synth q5_0 (S1). Measures:
(a) per-word error overlap with the soup: of the reference words the soup gets wrong, the share Parakeet gets right
(and the reverse); the same for DoRA vs soup as the within-Whisper baseline; (b) oracle PER/CER per clip over
{soup} vs {soup, Parakeet} vs {soup, DoRA, r16, synth} vs all five; (c) the reference-free medoid router over the
same sets (route.medoid, no confidence needed, since the two families' confidences are not on one scale).
Reading: Parakeet is complementary if (a) it rescues a larger share of the soup's errors than DoRA does, and (b)
adding it lowers the oracle by more than adding all three Whisper experts. Practical gain only if (c) medoid over a
set with Parakeet beats the soup alone by ≥ 0.5 PER points. Prediction: (a) and (b) yes (different architecture,
different errors); (c) no: with two strong-ish experts the medoid has no majority to work with, and Parakeet is
2.5× worse, so routing needs a confidence that knows when Parakeet is right.
Bar for any router (Ale, 2026-10-10): it counts only if it lowers WER, CER **and** PER against the best single
expert at once.

### M1 scored (offline, `scratchpad/m1.py` over the saved per-clip CSVs)
(a) Rescue: of the soup's 15 wrong reference words, **Parakeet gets 5 right (33 %)**, DoRA 2 (13 %). Of Parakeet's
33, the soup gets 23 right (70 %); of DoRA's 23, the soup gets 10 (43 %).
(b) Oracle (per clip, the member with the fewest phone errors):

| experts | WER | CER | PER | vowel |
|---|---|---|---|---|
| soup alone | 13.2 % | 3.5 % | 4.1 % | 4.8 % |
| soup + DoRA | 12.4 % | 2.6 % | 3.5 % | 4.4 % |
| soup + r16 | 10.7 % | 3.0 % | 3.2 % | 3.9 % |
| soup + aug-synth | 10.7 % | 2.5 % | 2.7 % | 3.5 % |
| **soup + Parakeet** | **10.7 %** | **1.9 %** | **2.4 %** | **3.1 %** |
| soup + 3 Whisper | 8.3 % | 1.2 % | 1.5 % | 2.6 % |
| all five | 7.4 % | 1.1 % | 1.4 % | 2.2 % |

(c) Medoid router: soup + 3 Whisper 3.8 % PER (WER 12.4, CER 3.3); all five 4.1 % (= the soup); 3 Whisper +
Parakeet 5.2 %. None lowers WER, CER and PER together below the soup.
Reading: (a) **yes**: Parakeet rescues 2.5× the share DoRA does, though it is the weakest single expert (CER 8.9 %
vs DoRA 5.6 %): different architecture, different errors. (b) as worded, **no** (adding three Whisper experts lowers
the oracle more than adding Parakeet), but the comparison was unfair (more members, more draws); at equal size
Parakeet is the best partner for the soup on every metric (pair oracle PER 2.4 % vs 2.7–3.5 % for any Whisper
partner). (c) **no**, as predicted: agreement voting cannot tell when the minority expert is right. The headroom is
real (soup + Parakeet oracle −1.7 PER, −1.6 CER, −2.5 WER points); realising it needs a confidence router fitted on
dev, not on these test clips. Prediction held on all three.

## 2026-10-10 — R1: a confidence router over soup + Parakeet (written before the dev decodes)
Experts: soup q8_0 (full window; confidence = mean token log-prob) and Parakeet run 3 (confidence = −TDT loss per
label token of its own hypothesis). The two confidences are on different scales, so each is z-scored with its mean
and SD over the 35 **dev** clips. Rules, all fitted on dev only, then scored once on the 39 test clips:
- **R1b (accuracy):** both decode; pick the expert with the higher z. No threshold.
- **R1c (latency):** Parakeet decodes; if its z ≥ t the final is Parakeet's, else the soup decodes and its text is
  the final. t from a grid on dev (the lowest dev PER; ties → the t that escalates fewer clips).
Bar (Ale): counts only if WER, CER **and** PER on test are all below the soup alone (13.2 / 3.5 / 4.1 %), and for
R1c also report the share of clips that reach the soup (latency cost).
Prediction: R1b lowers PER and CER by 0.3–0.8 points but not WER (Parakeet's wins are partial words: CER/PER, while
its whole-word errors are more numerous); R1c fails the bar: Parakeet alone is 2.5× worse, so it must escalate most
clips, and its confidence is unlikely to be sharp enough on 35 dev clips to keep the clips it gets right.

### D1 scored (`ahms:results/sonic/padtest_goodcores_*.json`, CPUs 2-5): the jitter was CPU 0
Raw input, 3 runs: **0/39** clips differ; 30 s pad, 2 runs: 0/39; raw vs padded: 0/39 differ; soup WER 13.2 %,
CER 3.5 %, PER 4.1 %, vowel 4.8 % every run. Prediction held. **S5a's explanation is withdrawn**: whisper.cpp is
deterministic on x86; the log-prob jitter, the word flips and the rule-2 failure came from the faulty core. The
padding stays (neutral) with a corrected comment. Consequence: kit run 3b (WER 9.9 %, PER 4.3 %) also ran with
CPU 0; the clean number for the soup + cascade + snap system is the M3's, WER 9.1 %, CER 2.4 %, PER 3.5 %,
vowel 4.8 % (S5/S5b, reproduced twice). Every ahms result before 2026-10-10 13:00 that used CPU 0 is suspect for
timing and for rare decode flips; the accuracy tables reproduced on the M3 or on good cores stand.

### R1 scored (`scratchpad/r1.py`; dev decodes `ahms:results/sonic/20261010-1324*`, CPUs 6-7)
Dev confidence: soup mean −0.062 (SD 0.071), Parakeet −0.129 (SD 0.130).

| system | dev WER / CER / PER / vowel | test WER / CER / PER / vowel |
|---|---|---|
| soup alone | 12.4 / 3.5 / 4.6 / 4.7 % | 13.2 / 3.5 / 4.1 / 4.8 % |
| Parakeet alone | 25.5 / 7.2 / 9.4 / 5.5 % | 28.9 / 8.9 / 11.9 / 8.7 % |
| R1b, higher z wins | 18.2 / 4.7 / 6.8 / 3.9 % (Parakeet on 19/35) | 13.2 / 3.3 / 4.0 / 3.9 % (Parakeet on 15/39) |
| R1c, dev-best t | t = +1.00 = always the soup | = soup |

Reading: **both fail the bar.** R1b is clearly worse than the soup on dev and only −0.2 CER / −0.1 PER on test with
WER unchanged; prediction (−0.3 to −0.8 on PER/CER) missed. R1c held its prediction (no t keeps Parakeet's wins).
Self-confidence of two different models does not say which is right: Parakeet is confident on clips the soup gets
right. The one consistent signal: vowel error drops on both splits (dev 4.7 → 3.9, test 4.8 → 3.9).

## 2026-10-10 — R2: cross-scoring the two hypotheses with both models (written before any number)
For each clip, two candidates: the soup's text and Parakeet's text. Each is scored by both models on the same
audio: Whisper soup (HF transformers, merged in memory as `soup.py` does, teacher-forced, mean token log-prob after
the `de / transcribe / notimestamps` prefix) and Parakeet run 3 (−TDT loss per token). Score(h) = z_W(h) + z_P(h),
each judge z-scored over its dev scores of both candidates; the higher score is the final. No threshold, nothing
else fitted. Secondary, reported but not the decision: the weight on Parakeet's judgement from a dev grid.
Reading: the same bar (WER, CER and PER all below the soup alone on test). Prediction: passes on CER and PER by
0.3–0.8 points, WER within one word of the soup; each judge prefers its own hypothesis, but on the clips where they
disagree the other judge's vote decides, which is what self-confidence (R1) could not do.

## 2026-10-10 — S6: Parakeet alone in the streaming harness (written before the run)
`local_decode.py` on the 39 clips with `SONIC_ENGINE=parakeet`, `SONIC_EXPERTS=models/parakeet-v3-christian-run3`
(one expert: it streams and it finalises; no cascade), `SONIC_PARTIAL_EVERY=0.25`, 4 threads, ahms CPUs 2-5
(2 P-cores with HT; x86, our share). LocalAgreement-2, energy gate, shedding and the speculative final as before.
Reading: latency against the board (takagi 0.40 + 0.15 s; JLShen 0.91 + 0.38 s) and rules 1-2; accuracy should
equal P2 (WER 28.9 %, CER 8.9 %) since the final is the same decode. Prediction: TTLT median ≤ 0.4 s (one decode
of the trimmed audio, often already done speculatively); TTFT-stable median 0.7–1.2 s (first word + two agreeing
decodes 0.25 s apart + decode cost + the 0.2 s gate); rules 1 and 2 pass. If so, the speed side of the phone-call
system is solved on 2 cores and the open problem is only accuracy (R2 / better adaptation).

### R2 scored (`sonic/xscore.py`, `sonic/xscore_eval.py`; `results/sonic/r2/`, GPU 1 + CPUs 6-7)
Bug found and fixed before reading: run 1 of the Whisper judge tokenised `" " + text` (training used no leading
space), so every first token was off and the soup's own text scored −0.89/token against whisper.cpp's −0.06. Fixed,
the judge matches whisper.cpp within ~0.006/token (8 dev clips: −0.014 vs −0.017 …); run 1 kept as
`judge_whisper_run1.csv`, not used.

| system | dev WER / CER / PER / vowel | test WER / CER / PER / vowel |
|---|---|---|
| soup alone | 12.4 / 3.5 / 4.6 / 4.7 % | 13.2 / 3.5 / 4.1 / 4.8 % |
| Parakeet alone | 25.5 / 7.2 / 9.4 / 5.5 % | 28.9 / 8.9 / 11.9 / 8.7 % |
| Whisper judge only | 11.7 / 3.4 / 4.4 / 4.7 % | 13.2 / 3.5 / 4.1 / 4.8 % (= soup) |
| Parakeet judge only | 24.8 / 6.8 / 9.0 / 5.1 % | 24.8 / 7.7 / 10.2 / 7.9 % |
| **R2, equal weights (the pre-registered rule)** | 13.1 / 3.3 / 4.3 / 2.4 % | **11.6 / 3.1 / 3.3 / 3.5 %** |
| R2, w_parakeet = 0.25 (chosen on dev, secondary) | 10.9 / 2.5 / 3.3 / 2.7 % | 12.4 / 3.3 / 4.0 / 4.4 % |

Reading: **R2 passes the bar on test** (WER −1.6, CER −0.4, PER −0.8 points; vowel −1.3) with the rule fixed in
advance. Prediction held (CER/PER −0.3 to −0.8; WER within a word: it is two words better). Caveat from dev: there
the same rule lowers CER and PER but not WER (+0.7 = one word), and the dev-chosen weight helps on dev but only
marginally on test. So: CER and PER improve on both splits; WER moves by one or two words either way, within this
test set's resolution (one word = 0.8 points). Each judge alone prefers its own model's text (Parakeet judge picks
Parakeet 34/35); only together do they decide. Cost: both experts decode, plus two scoring passes (~0.2 s for
Parakeet; Whisper teacher-forcing ~ one encoder pass).

### S6 scored (`ahms:results/sonic/kit_s6/`, CPUs 2-5 = 2 P-cores with HT, x86)
WER 28.9 %, CER 8.9 %, PER 11.9 %, vowel 8.7 % (= P2, as expected); **TTFT-stable median 1.12 s** (p90 2.89),
**TTLT median 0.30 s** (p90 0.50); rules 1 and 2 pass. Total 1.42 s (ranking latency, the mean, 0.71 s), against
the soup's 12.7 + 11.2 s (S5b, M3): ~17× faster on fewer, slower cores. Board for scale (other data, other task):
JLShen 0.91 + 0.38 s, takagi 0.40 + 0.15 s. Prediction held on both (TTLT ≤ 0.4 s; TTFT-stable 0.7–1.2 s).
Reading: the speed side of the live-call system works on 2 cores with re-decoding alone (no cache-aware encoder);
the open problem is accuracy at that latency. TTFT is now mostly LocalAgreement (two agreeing decodes 0.25 s apart)
plus the 0.2 s gate; the p90 comes from long clips, where each re-decode grows with the buffer.

## 2026-10-10 — P3: tuning the Parakeet fine-tune (written before any run)
Arms on GPU 1 / CPUs 2-7, same seed, same stopping rule (dev CER, patience 5, ≤ 40 epochs), test untouched:
A = run 3 (lr 1e-5, no augmentation; dev 7.5 %), B = lr 3e-5, C = A + augmentation (the Whisper recipe's
speed / low-pass / reverb / noise, p = 0.5 per effect), D = A with the encoder only trained, E = lr 3e-5 +
augmentation. Decision: the arm with the lowest best-dev CER replaces A only if it beats A's 7.5 % by ≥ 1.0 point
(dev moves ±1.5 epoch to epoch, so smaller is noise); only that one arm is scored on test, once, and in S6's
streaming harness. Prediction: augmentation helps most (it did for Whisper: the DoRA recipe uses 0.5), C or E
reaches dev 5–6 %; B converges faster but no better; D is worse (the joint and prediction networks must learn his
German text style too). On test the winner lands at CER 6–8 %.

### P3 scored (`ahms:results/sonic/p3/`, GPU 1 / CPUs 2-7)
Best dev CER per arm: A (run 3) 7.5 %, **B lr 3e-5 5.1 %**, C augmentation 8.5 %, **D encoder only 5.8 %**
(trains 609 of 627 M parameters: the prediction and joint networks are only 18 M), **E lr 3e-5 + augmentation
4.5 %** → E replaces A (−3.0 points, past the 1-point bar). B, C, D deleted (losers, reproducible from the seed).
E on test, once (`ahms:results/sonic/20261010-172143/`; streaming `ahms:results/sonic/kit_p3/`, CPUs 2-5):

| system | WER | CER | PER | vowel | TTFT-stable / TTLT (median) |
|---|---|---|---|---|---|
| soup q8_0 | 13.2 % | 3.5 % | 4.1 % | 4.8 % | 12.7 s / 11.2 s (M3, S5b) |
| Parakeet A (run 3) | 28.9 % | 8.9 % | 11.9 % | 8.7 % | 1.12 s / 0.30 s (S6) |
| **Parakeet E** | **13.2 %** | **3.1 %** | 4.9 % | **3.9 %** | **1.12 s / 0.29 s** (p90 2.47 / 0.49) |
| Parakeet E + snap | 11.6 % | 3.5 % | 5.3 % | 5.2 % | |

Rules 1 and 2 pass. Reading: prediction **beaten** (predicted test CER 6–8 %; got 3.1 %). The learning rate was the
main lever (B), augmentation helped only together with it (C alone was worse), so the first run was undertrained.
Parakeet E equals the soup on WER, beats it on CER (−0.4) and vowels (−0.9), trails on PER (+0.8), at ~1/40 of the
decode cost and inside a live-call latency on 2 cores. Not yet tuned further (lr 1e-4, more epochs, encoder-only
with lr 3e-5 + augmentation); and R2 should be rerun with E as the second expert.
