"""
Shared helpers for the Fine-Tuning Small AI Models workshop.

Everything the labs have in common lives here so each lab script stays short
and focused on the one idea it is teaching:

  * the prompt / completion format used for the triage task
  * loading the ticket dataset through Hugging Face `datasets`
  * generating predictions from any causal LM
  * scoring predictions against the required schema

You do not need to edit this file during the workshop, but reading it is
worthwhile -- it is where the "boring" plumbing lives.
"""

import json
import os
import re
import time

import torch
from datasets import load_dataset

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "out")
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# The task: turn a free-text support ticket into one strict schema line.
# --------------------------------------------------------------------------

CATEGORIES = [
    "access-management",
    "billing-inquiry",
    "data-request",
    "hardware-failure",
    "network-connectivity",
    "onboarding-request",
    "security-incident",
    "software-bug",
]
PRIORITIES = ["P1", "P2", "P3", "P4"]
QUEUES = [
    "app-support",
    "endpoint-ops",
    "finance-ops",
    "identity-team",
    "network-ops",
    "security-ops",
]

PROMPT_TEMPLATE = "### Ticket:\n{ticket}\n\n### Triage:\n"

# The best-case prompt for a model we are NOT fine-tuning: hand it the entire
# label space and the exact output contract. Lab 5 uses this to give a frontier
# model its fairest possible shot at the task.
ZERO_SHOT_SYSTEM = f"""You are a support ticket triage system for Northwind Logistics.

Read the ticket and reply with EXACTLY ONE LINE in this format, nothing else:
category: <category> | priority: <priority> | queue: <queue>

Valid categories: {", ".join(CATEGORIES)}
Valid priorities: P1 (outage or security incident), P2 (a team is blocked), \
P3 (normal request), P4 (low urgency)
Valid queues: {", ".join(QUEUES)}

Queue is determined by category:
  access-management, onboarding-request -> identity-team
  hardware-failure                      -> endpoint-ops
  network-connectivity                  -> network-ops
  software-bug, data-request            -> app-support
  billing-inquiry                       -> finance-ops
  security-incident                     -> security-ops

Priority comes from the urgency language in the ticket, not from the category.
Output the single line only. No explanation, no preamble, no code fences."""

# A prediction only counts as schema-valid if it matches this exactly.
SCHEMA_RE = re.compile(
    r"^category:\s*(?P<category>[a-z-]+)\s*\|\s*"
    r"priority:\s*(?P<priority>P[1-4])\s*\|\s*"
    r"queue:\s*(?P<queue>[a-z-]+)\s*$"
)


def build_prompt(ticket: str) -> str:
    """Wrap a raw ticket in the instruction format the model is trained on."""
    return PROMPT_TEMPLATE.format(ticket=ticket)


def load_tickets():
    """Load the ticket dataset with the Hugging Face `datasets` library.

    Returns a DatasetDict with 'train' and 'eval' splits. Each row has:
    ticket, label, category, priority, queue.
    """
    return load_dataset(
        "json",
        data_files={
            "train": os.path.join(DATA_DIR, "tickets_train.jsonl"),
            "eval": os.path.join(DATA_DIR, "tickets_eval.jsonl"),
        },
    )


def to_prompt_completion(example):
    """Map a raw row to the prompt/completion format TRL understands.

    TRL detects the 'prompt' and 'completion' columns and automatically masks
    the prompt tokens out of the loss -- the model is only graded on the part
    we actually want it to produce.
    """
    return {
        "prompt": build_prompt(example["ticket"]),
        "completion": example["label"],
    }


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def generate_predictions(model, tokenizer, rows, max_new_tokens=24, show=0):
    """Run greedy generation over `rows` and return the raw predicted strings.

    Greedy (do_sample=False) because this is a classification-style task: we
    want the single most likely triage line, not a creative one.
    """
    model.eval()
    preds = []
    start = time.time()
    for i, row in enumerate(rows):
        prompt = build_prompt(row["ticket"])
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                # Belt and braces: a small model that has not fully learned to
                # emit EOS will still be cut off at the end of its first line.
                stop_strings=["\n"],
                tokenizer=tokenizer,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        # Keep only the first line -- anything after it is the model rambling.
        pred = text.strip().split("\n")[0].strip()
        preds.append(pred)
        if i < show:
            print(f"\n  TICKET   : {row['ticket'][:90]}...")
            print(f"  EXPECTED : {row['label']}")
            print(f"  PREDICTED: {pred!r}")
    print(f"\n  ({len(rows)} predictions in {time.time() - start:.1f}s)")
    return preds


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def score(preds, rows):
    """Score predictions against the gold labels.

    Returns a dict with four numbers:
      schema_valid  -- fraction that match the required schema at all
      category_acc  -- fraction with the correct category
      priority_acc  -- fraction with the correct priority
      queue_acc     -- fraction with the correct queue
    """
    n = len(rows)
    valid = cat = pri = que = 0
    for pred, row in zip(preds, rows):
        m = SCHEMA_RE.match(pred)
        if not m:
            continue
        valid += 1
        cat += int(m.group("category") == row["category"])
        pri += int(m.group("priority") == row["priority"])
        que += int(m.group("queue") == row["queue"])
    return {
        "n": n,
        "schema_valid": valid / n,
        "category_acc": cat / n,
        "priority_acc": pri / n,
        "queue_acc": que / n,
    }


def print_scores(name, s):
    print(f"\n  === {name} (n={s['n']}) ===")
    print(f"    schema valid : {s['schema_valid']:6.1%}")
    print(f"    category acc : {s['category_acc']:6.1%}")
    print(f"    priority acc : {s['priority_acc']:6.1%}")
    print(f"    queue acc    : {s['queue_acc']:6.1%}")


def save_result(key, scores, extra=None):
    """Append one run's scores to out/results.json so Lab 5 can compare them."""
    path = os.path.join(OUT_DIR, "results.json")
    results = {}
    if os.path.exists(path):
        with open(path) as f:
            results = json.load(f)
    results[key] = {**scores, **(extra or {})}
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved '{key}' to {path}")
    return results


# --------------------------------------------------------------------------
# Small conveniences
# --------------------------------------------------------------------------

def param_report(model, label="model"):
    """Print total vs. trainable parameter counts -- the LoRA money shot."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  {label}: {trainable:,} trainable / {total:,} total "
          f"({100 * trainable / total:.4f}%)")
    return trainable, total


def model_size_mb(model):
    """Approximate in-memory size of the model's parameters, in MB."""
    total = 0
    for p in model.parameters():
        # 4-bit weights are stored packed in uint8, so element_size() is 1
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total / 1024 ** 2


# --------------------------------------------------------------------------
# Optional: a frontier-model baseline via Groq
# --------------------------------------------------------------------------

# Groq retires hosted models on a regular cadence. The model this workshop
# originally used, `llama-3.3-70b-versatile`, was DECOMMISSIONED on 2026-08-16.
# If this step suddenly 404s, that is almost certainly why: check
# https://console.groq.com/docs/deprecations and change the two constants below.
# Nothing else in the workshop depends on them.
#
# That churn is not a footnote -- it is one of the arguments for fine-tuning a
# model you own, and Lab 5 Step 6 makes the point explicitly.
FRONTIER_MODEL = "openai/gpt-oss-120b"   # 120B MoE (~5B active per token).
                                         # Groq's own recommended replacement.

# gpt-oss models ALWAYS reason -- reasoning cannot be switched off, only turned
# down. Two consequences for a 1-line classification task:
#   reasoning_effort="low"     keeps the thinking short
#   reasoning_format="hidden"  keeps it out of the reply, so `content` is just
#                              the triage line
# And reasoning tokens are drawn from the OUTPUT budget, so the 48 tokens this
# task needs is no longer a safe cap: too small a budget gets spent thinking and
# returns an EMPTY content string with finish_reason="length".
FRONTIER_KWARGS = {
    "reasoning_effort": "low",
    "reasoning_format": "hidden",
    "max_completion_tokens": 256,
}


def groq_predictions(rows, model=FRONTIER_MODEL, show=0, **model_kwargs):
    """Zero-shot predictions from a large hosted model, for comparison.

    Returns None if the groq package is missing, GROQ_API_KEY is not set, or the
    API call fails -- this is an optional step and every lab must run without it.
    """
    if not os.environ.get("GROQ_API_KEY"):
        return None
    try:
        from groq import Groq
    except ImportError:
        print("  (groq package not installed -- pip install groq)")
        return None

    # The SDK already retries 429s on its own (twice by default), honouring the
    # Retry-After header. We raise that a little because a whole classroom hits
    # the free tier's 30 requests/minute at the same moment.
    client = Groq(max_retries=5)
    kwargs = {**FRONTIER_KWARGS, **model_kwargs}

    preds, empties = [], 0
    start = time.time()
    for i, row in enumerate(rows):
        messages = [
            {"role": "system", "content": ZERO_SHOT_SYSTEM},
            {"role": "user", "content": row["ticket"]},
        ]
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                temperature=0,          # greedy, like our local models
                **kwargs,
            )
        except Exception as e:
            # A non-reasoning model will reject the reasoning_* arguments. Drop
            # them and try once more, so swapping FRONTIER_MODEL stays a
            # one-line change.
            if type(e).__name__ == "BadRequestError" and any(
                    k.startswith("reasoning") for k in kwargs):
                kwargs = {k: v for k, v in kwargs.items()
                          if not k.startswith("reasoning")}
                print("  (model rejected the reasoning_* args -- retrying without them)")
                try:
                    resp = client.chat.completions.create(
                        model=model, messages=messages, temperature=0, **kwargs)
                except Exception as e2:
                    print(f"  Groq call failed on row {i}: {e2}")
                    return None
            else:
                print(f"  Groq call failed on row {i}: {e}")
                print(f"  If this is a 404, '{model}' may have been retired --")
                print("  see https://console.groq.com/docs/deprecations")
                return None

        text = resp.choices[0].message.content or ""
        if not text.strip():
            empties += 1
        pred = text.strip().split("\n")[0].strip().strip("`").strip()
        preds.append(pred)
        if i < show:
            print(f"\n  TICKET   : {row['ticket'][:90]}...")
            print(f"  EXPECTED : {row['label']}")
            print(f"  PREDICTED: {pred!r}")

    print(f"\n  ({len(rows)} predictions from {model} in {time.time() - start:.1f}s)")
    if empties:
        print(f"  WARNING: {empties} of {len(rows)} replies came back EMPTY.")
        print("  That is the reasoning-token trap: raise max_completion_tokens")
        print("  in FRONTIER_KWARGS (common/triage.py) and rerun.")
    return preds
