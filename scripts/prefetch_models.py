"""Download every model and dataset the labs use, so no lab waits on a download.

    python scripts/prefetch_models.py

Cross-platform version of scripts/prefetch_models.sh, for running the labs on
your own machine (macOS, Windows, Linux). About 4 GB in total. Safe to re-run:
anything already downloaded is skipped.

Files go to the standard Hugging Face cache, which is exactly where the lab
scripts look for them:
    macOS / Linux : ~/.cache/huggingface
    Windows       : C:\\Users\\<you>\\.cache\\huggingface
(If you have set HF_HOME yourself, that location is used instead.)

No Hugging Face account or token is needed - every model and dataset here is
public.
"""
import os
import sys

MODELS = [
    ("distilgpt2", "Labs 1-2, ~350 MB"),
    ("HuggingFaceTB/SmolLM2-360M-Instruct", "Labs 1, 3, 4, 5, ~1.4 GB"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Lab 4 Step 8, ~2.2 GB"),
]

DATASETS = [
    ("allenai/ai2_arc", "ARC-Easy"),
    ("allenai/ai2_arc", "ARC-Challenge"),
]


def main():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import constants
    except ImportError:
        print("The training libraries are not installed in this Python environment.")
        print("Activate your .venv first, then: pip install -r requirements.txt")
        return 1

    print(f">>> Downloading workshop models into {constants.HF_HOME}")
    for name, note in MODELS:
        print(f"  fetching {name}  ({note}) ...", flush=True)
        AutoTokenizer.from_pretrained(name)
        AutoModelForCausalLM.from_pretrained(name)
        print(f"  done {name}", flush=True)

    print(">>> Downloading the evaluation datasets used by lm-evaluation-harness (Lab 5)")
    try:
        from datasets import load_dataset
        for repo, config in DATASETS:
            load_dataset(repo, config, split="test")
        print("  eval datasets cached")
    except Exception as e:  # Lab 5 will download them itself if this fails
        print(f"  (skipped eval dataset download: {e})")

    print(">>> Model prefetch complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
