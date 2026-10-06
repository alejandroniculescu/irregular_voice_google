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
