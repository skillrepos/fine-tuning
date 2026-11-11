#!/usr/bin/env bash
# Download every model the labs use, at container-build time, so that no lab
# step during the live workshop is waiting on a ~2 GB download.
#
# Safe to re-run: the Hugging Face cache skips anything already present.
set -e

if [ -d ".venv" ]; then
    source ./.venv/bin/activate
fi

export HF_HOME="${HF_HOME:-$(pwd)/.cache/huggingface}"

echo ">>> Pre-fetching workshop models into $HF_HOME"
python3 - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    "distilgpt2",                            # Lab 1, Lab 2   (~350 MB)
    "HuggingFaceTB/SmolLM2-360M-Instruct",   # Lab 3, Lab 4   (~1.4 GB)
]

for m in MODELS:
    print(f"  fetching {m} ...", flush=True)
    AutoTokenizer.from_pretrained(m)
    AutoModelForCausalLM.from_pretrained(m)
    print(f"  done {m}", flush=True)
PY

echo ">>> Pre-fetching the evaluation datasets used by lm-evaluation-harness"
python3 - <<'PY'
try:
    from datasets import load_dataset
    load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    print("  eval datasets cached")
except Exception as e:
    print(f"  (skipped eval dataset prefetch: {e})")
PY

echo ">>> Model prefetch complete."
