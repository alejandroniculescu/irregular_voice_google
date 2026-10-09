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
