# SAP corpus access: one-page proposal (draft 2026-10-09, for Ale to edit and send)

To: speechaccessibility@beckman.illinois.edu — subject: SAP corpus DUA request, SAPC2 Track 2 / personalised streaming ASR

**Applicant.** Alexander Niculescu, research associate, FG AHMS, TU Berlin (PhD from autumn 2026). Open-source
project: https://github.com/alejandroniculescu/irregular_voice_google (Apache 2.0).

**Purpose.** Participation in the Speech Accessibility Project Challenge 2 (Track 2, streaming ASR) and a study of
what per-speaker adaptation adds on top of a speaker-independent streaming system for dysarthric speech.

**What we have done.** A personalised German ASR for one speaker with dysarthria: a Whisper large-v3-turbo
adapter (DoRA, rank 32) trained on 12 minutes of his speech, plus a deterministic sound-based correction step and a
guard that never acts on an uncertain command. On his held-out recordings, word error rate falls from 48.8 % (base
model) to 13.2 %; a 15-command replay produced 0 wrong actions. Everything runs on-device (whisper.cpp, CPU).
We also found that his adapter does not transfer to other speakers with dysarthria (125 recordings, 60.6 % → 109.4 %
WER), which is the question the SAP corpus lets us study properly.

**What we would do with the corpus.**
1. Build a Track 2 streaming system under the challenge's reject policy (stable partials, no early guessing,
   deterministic two-pass output), CPU-only, from a streaming-native base.
2. Measure, per SAP speaker, what a small per-speaker adapter trained on a few minutes of that speaker adds to the
   speaker-independent system, as a function of minutes of adaptation audio and of etiology.
3. Report CER/WER with the challenge's two-reference scoring and the latency metrics (TTFT-stable, TTLT).

**Data handling.** Audio stays on one lab GPU machine (TU Berlin) and the applicant's encrypted laptop; no copies to
cloud services; no redistribution; derived models are not released if they can reproduce a speaker's voice;
results reported in aggregate per etiology, never per identifiable speaker; data deleted at the end of the agreed
period. We will sign the Data Transfer and Use Agreement as provided.

**Timeline.** DUA on receipt; system build October–November 2026; results to the SAPC2 workshop (NeurIPS 2026) if
the timeline allows, otherwise to the next edition and a journal article.
