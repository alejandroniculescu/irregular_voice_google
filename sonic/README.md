# sonic: the hackathon build

Everything aimed at speed and at winning SAPC2 Track 2 (streaming, CPU, reject policy; deadline 2026-10-24) lives
here, apart from the research code in `src/irregular_voice_google`. Code here may import from the package; the
package never imports from here.

| file | what |
|---|---|
| `route.py` | reference-free routers over several adapters' hypotheses (medoid, ROVER, conservative switch); reports WER, CER, PER, vowel error rate |

Rules carry over: Christian's audio and transcripts stay on the Mac and ahms; outputs go to git-ignored `results/`.
