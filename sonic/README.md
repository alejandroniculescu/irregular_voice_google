# sonic: the hackathon build

Everything aimed at speed and at winning SAPC2 Track 2 (streaming, CPU, reject policy; deadline 2026-10-24) lives
here, apart from the research code in `src/irregular_voice_google`. Code here may import from the package; the
package never imports from here.

| file | what |
|---|---|
| `route.py` | reference-free routers over several adapters' hypotheses (medoid, ROVER, conservative switch); reports WER, CER, PER, vowel error rate |
| `engine.py` | whisper.cpp in-process on CPU: one model per expert, loaded once; encoder window, token cap, mean log-prob |
| `bench.py` | CPU bench of experts × encoder windows and the routers over them (S1) |
| `track2/model.py` | the SAPC2 Track 2 submission: streaming partials (LocalAgreement, energy gate) + confidence-cascade final over the experts |
| `track2/make_manifest.py` | the kit's manifests for our test clips (git-ignored output) |
| `track2/score.py` | accuracy + the kit's latency metrics and reject rules for a `local_decode.py` run |

The story the build shows: **streaming** (partials every 0.5 s, a word shown once two decodes agree), a **mixture of
experts** (several personal adapters, merged into whisper.cpp models) and **routing** (the first expert answers
when confident, otherwise every expert decodes and the most confident wins).

    uv venv .venv-sonic --python 3.12 && uv pip install --python .venv-sonic/bin/python pywhispercpp numpy jiwer gruut gruut_lang_de wordfreq pytest
    PYTHONPATH=src:. .venv-sonic/bin/python -m sonic.bench --ctx 0 512
    .venv-sonic/bin/python sonic/track2/make_manifest.py
    .venv-sonic/bin/python data/ext/SAPC-template/track2_starting_kit/local_decode.py --submission-dir sonic/track2 \
        --manifest-csv results/sonic/kit/Test.csv --streaming-manifest-csv results/sonic/kit/Test_streaming.csv \
        --data-root data/processed/trim --out-csv results/sonic/kit/Test.predict.csv \
        --out-partial-json results/sonic/kit/Test.partial_results.json
    PYTHONPATH=src:. .venv-sonic/bin/python sonic/track2/score.py results/sonic/kit

The kit is cloned (not vendored) into git-ignored `data/ext/SAPC-template`.

Rules carry over: Christian's audio and transcripts stay on the Mac and ahms; outputs go to git-ignored `results/`.
