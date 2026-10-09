"""Probe why the sound-gated LLM fixer changed nothing (E3a, 2026-10-09). Prints COUNTS only, never text.

    .venv/bin/python scripts/llmfix_probe.py results/<ts>/models__lora-dora-r32+snap.csv qwen3.5:4b qwen3.5:9b gemma4:26b

For each hypothesis with at least one word error (hypothesis != reference after normalisation) and each model:
the raw Ollama response (with and without thinking disabled), classified as empty / identical to input / a
one-for-one swap proposal / something else (e.g. reasoning text, multi-line), and whether the gate kept a swap.
"""
import csv, json, sys, urllib.request
from irregular_voice_google.llmfix import PROMPT, OLLAMA, gate, WORD
from irregular_voice_google import snap as snapmod
from irregular_voice_google.text import normalize


def call(text, model, think):
    body = {"model": model, "prompt": PROMPT.format(text=text), "stream": False,
            "options": {"temperature": 0, "num_predict": 200}}
    if think is not None:
        body["think"] = think
    req = urllib.request.Request(OLLAMA, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    return d.get("response", ""), d.get("thinking", "")


def classify(text, resp):
    if not resp.strip():
        return "empty"
    first = resp.strip().splitlines()[0].strip().strip('"„“')
    if normalize(first) == normalize(text):
        return "identical"
    a, b = WORD.findall(text), WORD.findall(first)
    if len(a) == len(b):
        return "one_for_one_candidate"
    return "other"


def main():
    csv_path, models = sys.argv[1], sys.argv[2:]
    rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8")) if normalize(r["reference"]) != normalize(r["hypothesis"])]
    snapper = snapmod.for_speaker("data/processed/trim/manifest.csv", "resources/lexicon")
    print(f"{len(rows)} hypotheses with errors")
    for model in models:
        for think in (None, False):
            counts = {"empty": 0, "identical": 0, "one_for_one_candidate": 0, "other": 0, "gate_kept_swap": 0, "had_thinking_field": 0, "errors": 0}
            for r in rows:
                try:
                    resp, thinking = call(r["hypothesis"], model, think)
                except Exception:  # noqa: BLE001
                    counts["errors"] += 1; continue
                counts["had_thinking_field"] += bool(thinking)
                c = classify(r["hypothesis"], resp); counts[c] += 1
                if c == "one_for_one_candidate":
                    first = resp.strip().splitlines()[0].strip().strip('"„“')
                    if gate(r["hypothesis"], first, snapper) != r["hypothesis"]:
                        counts["gate_kept_swap"] += 1
            print(f"{model:<14} think={think!s:<5} {counts}")


if __name__ == "__main__":
    main()
