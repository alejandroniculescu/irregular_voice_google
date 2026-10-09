# Handoff: cereSprache (this repo), state on 2026-10-09 evening

You are the session working on cereSprache, the personal German ASR for one speaker with dysarthria (Christian).
The Parkinson's work (cereVox) lives in ~/dev/cerefox and has its own session; don't touch it.

## Read first
- `docs/open_questions.md` — every experiment, its pre-written reading rule and its result (E1, E3a, PER, SAPC2).
- `README.md`, `docs/USAGE.md`.

## Where things stand
- Best system: DoRA r32 (`models/lora-dora-r32` on ahms). Test (39 clips, 121 words, 657 phones): WER 13.2 %,
  CER 4.1 %, **PER 5.8 %** after snap. PER is the primary metric (`ivg-per`, gruut German IPA).
- Closed: generic-voice synthetic data (no evidence); NeuTTS cloned-voice data (hurts consonants); local LLM fixer
  (hurts; qwen3.5:9b is the chosen model, opt-in only, `think: false` is required).
- Oracle over adapters: PER 1.7 % if a router picked the best adapter per utterance → E2 (mixture of adapters) is
  the open lever; it needs new recordings.
- Target: SAPC2 Track 2 (streaming, CPU, reject policy). Deadline 24 Oct 2026. SAP corpus is US English and needs a
  DUA (draft: `docs/sapc2_proposal_draft.md`; Ale sends it).

## Next task (agreed, not started)
SAPC2 step 1, no new data needed: wrap the current pipeline (whisper.cpp + adapter) in the kit's `Model` interface
(https://github.com/xiuwenz2/SAPC-template, `track2_starting_kit`: `__init__`, `set_partial_callback`, `reset`,
`accept_chunk(100 ms float32 16 kHz) -> str`, `input_finished() -> str`); run `local_decode.py` and
`utils/track2_reject_check.py` on Christian's 39 test clips; measure TTFT-stable and TTLT on CPU. If rule 1
(unstable partials) fails, add LocalAgreement (emit only the prefix two consecutive decodes agree on). Only then
consider a streaming-native model.

## Rules that carry over
- His audio and transcripts never leave the Mac/ahms; `results/` and `models/` are git-ignored. Never paste
  transcripts into reports or commits.
- ahms: reach it as `ssh ahms-own`. Repo there: `/home/alex/irregular_voice_google` (old git, code deployed as
  uncommitted files); venvs `.venv` (main), `.venv-neutts` (TTS only). GPUs are shared; disk is 96 % full, pull
  nothing big. On the Mac use `.venv-compare` (numpy, jiwer, wordfreq, gruut, pytest) with `PYTHONPATH=src`.
- Write the reading rule and prediction into `docs/open_questions.md` before running; identical outputs across
  arms are a bug until shown otherwise.
- Commit and push when Ale says so; end commit messages with the Co-Authored-By line the harness gives you.
