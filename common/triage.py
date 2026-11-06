"""
Shared helpers for the Fine-Tuning Small AI Models workshop.

Everything the labs have in common lives here so each lab script stays short
and focused on the one idea it is teaching:

  * the prompt / completion format used for the triage task
  * loading the ticket dataset through Hugging Face `datasets`
  * getting answers from any model that writes text
  * scoring those answers against the correct labels

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
# The task: turn a support ticket into one line in a fixed format.
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

# The best prompt we can write for a model we are NOT fine-tuning: it lists
# every category, priority and queue, and the exact answer format. Lab 1 gives
# it (plus one worked example per category) to SmolLM2 before any training;
# Lab 5 gives it to a large hosted model. Both are the fairest shot a prompt
# can get.
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

# An answer only counts as "right format" if it matches this exactly.
SCHEMA_RE = re.compile(
    r"^category:\s*(?P<category>[a-z-]+)\s*\|\s*"
    r"priority:\s*(?P<priority>P[1-4])\s*\|\s*"
    r"queue:\s*(?P<queue>[a-z-]+)\s*$"
)

# What you would score by picking a category, priority and queue at random,
# on average: 1 in 8, 1 in 4 and 1 in 6. Any model worth using should beat it.
RANDOM_GUESS = {
    "category_acc": 1 / len(CATEGORIES),
    "priority_acc": 1 / len(PRIORITIES),
    "queue_acc": 1 / len(QUEUES),
}


def build_prompt(ticket: str) -> str:
    """Wrap a raw ticket in the instruction format the model is trained on."""
    return PROMPT_TEMPLATE.format(ticket=ticket)


def load_tickets():
    """Load the ticket dataset with the Hugging Face `datasets` library.

    Returns the data in two parts, 'train' and 'eval'. Each row has:
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
    """Split a raw row into the two parts the labs train on.

    'prompt' is the ticket wrapped in PROMPT_TEMPLATE; 'completion' is the
    label the model must produce. Keeping them separate is what lets Lab 3
    grade the model on the completion only, not on repeating the ticket.
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
                # A backup: a small model that hasn't fully learned to stop
                # (by writing its end token) still gets cut off after one line.
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


def pick_examples(train_rows, k=3):
    """Pick k worked examples from the training data, each from a different category."""
    examples, seen = [], set()
    for row in train_rows:
        if len(examples) >= k:
            break
        if row["category"] not in seen:
            examples.append(row)
            seen.add(row["category"])
    return examples


def best_prompt_messages(ticket, examples):
    """The best prompt we can give a chat model without training it.

    The full instructions (ZERO_SHOT_SYSTEM), then each worked example as a
    ticket and its correct answer, then the new ticket.
    """
    messages = [{"role": "system", "content": ZERO_SHOT_SYSTEM}]
    for ex in examples:
        messages.append({"role": "user", "content": ex["ticket"]})
        messages.append({"role": "assistant", "content": ex["label"]})
    messages.append({"role": "user", "content": ticket})
    return messages


def best_prompt_predictions(model, tokenizer, rows, examples, max_new_tokens=40, show=0):
    """Answers from an untrained chat model, given the best prompt we can write.

    Unlike generate_predictions(), this keeps the whole answer, not just the
    first line, so an answer spread over several lines can still earn credit
    for the parts it gets right.
    """
    model.eval()
    preds = []
    start = time.time()
    for i, row in enumerate(rows):
        text = tokenizer.apply_chat_template(
            best_prompt_messages(row["ticket"], examples),
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        pred = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()
        preds.append(pred)
        if i < show:
            print(f"\n  TICKET   : {row['ticket'][:90]}...")
            print(f"  EXPECTED : {row['label']}")
            print(f"  PREDICTED: {pred!r}")
    print(f"\n  ({len(rows)} predictions in {time.time() - start:.1f}s, "
          f"prompt about {inputs['input_ids'].shape[1]} tokens each)")
    return preds


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _find_label(text, field, choices):
    """Find the value the answer gives for one field (category, priority, queue).

    First look for the field written with its name, like "Category: billing"
    -- in any case, on any line, in any order. A shortened name counts if it
    can only mean one choice ("billing" -> "billing-inquiry"). If the field
    name is missing, take the first exact choice that appears anywhere.
    Returns None when the answer gives no usable value.
    """
    def norm(s):
        return re.sub(r"[\s_]+", "-", s.strip().lower())

    m = re.search(rf"{field}\s*[:=]\s*([A-Za-z0-9_ -]+)", text, re.IGNORECASE)
    if m:
        given = norm(m.group(1))
        for c in choices:
            if given == c.lower() or given.startswith(c.lower() + "-"):
                return c
        starts = [c for c in choices if c.lower().startswith(given)]
        if given and len(starts) == 1:
            return starts[0]

    flat = norm(text)
    hits = []
    for c in choices:
        m = re.search(rf"(?<![a-z0-9]){re.escape(c.lower())}(?![a-z0-9])", flat)
        if m:
            hits.append((m.start(), c))
    return min(hits)[1] if hits else None


def extract_fields(pred):
    """Pull category, priority and queue out of an answer, even a messy one."""
    m = SCHEMA_RE.match(pred.strip().split("\n")[0].strip())
    if (m and m.group("category") in CATEGORIES
            and m.group("queue") in QUEUES):
        return m.groupdict()
    return {
        "category": _find_label(pred, "category", CATEGORIES),
        "priority": _find_label(pred, "priority", PRIORITIES),
        "queue": _find_label(pred, "queue", QUEUES),
    }


def score(preds, rows):
    """Score the model's answers against the correct answers.

    Returns a dict with four fractions (printed with the plain labels below):
      schema_valid  -- "right format": the first line is exactly
                       category: ... | priority: ... | queue: ...
      category_acc  -- "category correct"
      priority_acc  -- "priority correct"
      queue_acc     -- "queue correct"
    The three "correct" scores do NOT require the right format. If the answer
    says "Category: billing-inquiry" on its own line, or in another order, the
    category still counts. Format is scored separately, by schema_valid.
    """
    n = len(rows)
    valid = cat = pri = que = 0
    for pred, row in zip(preds, rows):
        valid += int(bool(SCHEMA_RE.match(pred.strip().split("\n")[0].strip())))
        got = extract_fields(pred)
        cat += int(got["category"] == row["category"])
        pri += int(got["priority"] == row["priority"])
        que += int(got["queue"] == row["queue"])
    return {
        "n": n,
        "schema_valid": valid / n,
        "category_acc": cat / n,
        "priority_acc": pri / n,
        "queue_acc": que / n,
    }


def print_scores(name, s):
    print(f"\n  === {name} (n={s['n']}) ===")
    print(f"    right format     : {s['schema_valid']:6.1%}")
    print(f"    category correct : {s['category_acc']:6.1%}")
    print(f"    priority correct : {s['priority_acc']:6.1%}")
    print(f"    queue correct    : {s['queue_acc']:6.1%}")


def print_random_guess():
    """Print the random-guessing scores in the same layout as print_scores."""
    print("\n  === RANDOM GUESSING (average, no model at all) ===")
    print("    right format     :      -")
    print(f"    category correct : {RANDOM_GUESS['category_acc']:6.1%}")
    print(f"    priority correct : {RANDOM_GUESS['priority_acc']:6.1%}")
    print(f"    queue correct    : {RANDOM_GUESS['queue_acc']:6.1%}")


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
    print(f"\n  {label}: {trainable:,} of {total:,} numbers can be trained "
          f"({100 * trainable / total:.4f}%)")
    return trainable, total


def plain_progress(trainer):
    """Swap the Trainer's progress bar and raw log lines for one plain line
    every time it reports the loss."""
    from transformers import TrainerCallback
    from transformers.trainer_callback import PrinterCallback, ProgressCallback

    trainer.remove_callback(ProgressCallback)
    trainer.remove_callback(PrinterCallback)

    class PlainProgress(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            print(f"\n  Training for {state.max_steps} steps. The loss is how wrong the "
                  "model's guesses are\n  -- watch it go down.\n", flush=True)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                print(f"    step {state.global_step:>3} of {state.max_steps}   "
                      f"loss {float(logs['loss']):.3f}", flush=True)

    trainer.add_callback(PlainProgress())


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
# Optional: compare with a huge hosted model on Groq
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
    """Answers from a large hosted model, prompt only (no training), for comparison.

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
                print("  (this model did not accept some settings -- retrying without them)")
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
        print("  The model probably used up its reply length thinking before it")
        print("  answered. Raise max_completion_tokens in FRONTIER_KWARGS")
        print("  (common/triage.py) and rerun.")
    return preds
