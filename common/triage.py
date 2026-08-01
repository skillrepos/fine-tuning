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

def groq_predictions(rows, model="llama-3.3-70b-versatile", show=0):
    """Zero-shot predictions from a large hosted model, for comparison.

    Returns None if the groq package is missing or GROQ_API_KEY is not set --
    this is an optional step and every lab must run without it.
    """
    if not os.environ.get("GROQ_API_KEY"):
        return None
    try:
        from groq import Groq
    except ImportError:
        print("  (groq package not installed -- pip install groq)")
        return None

    client = Groq()
    preds = []
    start = time.time()
    for i, row in enumerate(rows):
        text = ""
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": ZERO_SHOT_SYSTEM},
                        {"role": "user", "content": row["ticket"]},
                    ],
                    temperature=0,        # greedy, like our local models
                    max_tokens=48,
                )
                text = resp.choices[0].message.content or ""
                break
            except Exception as e:
                # 30 requests/minute on the free tier -- back off and retry.
                if "429" in str(e) and attempt < 3:
                    time.sleep(3 * (attempt + 1))
                    continue
                print(f"  Groq call failed on row {i}: {e}")
                return None
        pred = text.strip().split("\n")[0].strip().strip("`").strip()
        preds.append(pred)
        if i < show:
            print(f"\n  TICKET   : {row['ticket'][:90]}...")
            print(f"  EXPECTED : {row['label']}")
            print(f"  PREDICTED: {pred!r}")
    print(f"\n  ({len(rows)} predictions from {model} in {time.time() - start:.1f}s)")
    return preds
