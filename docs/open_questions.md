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
