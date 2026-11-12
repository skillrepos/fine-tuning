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
import sys
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

def generate_predictions(model, tokenizer, rows, max_new_tokens=24, show=0,
                         on_answer=None):
    """Get the model's answer to every row, as plain text.

    do_sample=False makes the model always pick its most likely next word, so
    we get its single best triage line, not a creative one.

    on_answer, if given, is called with the answers so far after each one
    (the labs use it to fill in a file while the model works).
    """
    model.eval()
    preds = []
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
        if on_answer:
            on_answer(preds)
        if i < show:
            print(f"\n  TICKET   : {row['ticket'][:70]}...")
            print(f"  EXPECTED : {row['label']}")
            print(f"  GOT      : {pred[:70]!r}")
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


def right_format(answer):
    """True if the answer's first line is exactly
    category: <category> | priority: <priority> | queue: <queue>"""
    return bool(SCHEMA_RE.match(answer.strip().split("\n")[0].strip()))


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
        valid += int(right_format(pred))
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


def write_answer_log(path, rows, columns, title=None, only_mistakes=False):
    """Write every test ticket, its correct answer, and each model's answer
    so far to a text file. Rewritten after every answer, so the file fills in
    while the lab runs.

    title may be several lines (joined with \n).
    columns is a list of (name, answers) pairs; answers may be shorter than
    rows while a model is still working. With only_mistakes=True, a ticket is
    listed only once the first column has answered it with a field wrong.
    """
    fields = ("category", "priority", "queue")
    width = max(len("correct answer"), *(len(name) for name, _ in columns))
    lines = (title or "Each test ticket, its correct answer, and what each model wrote.").split("\n")
    lines += ["This file fills in as the lab runs.  \u2713 = right   \u2717 = wrong", ""]
    for i, row in enumerate(rows):
        if only_mistakes:
            first = columns[0][1]
            if i >= len(first):
                continue
            got = extract_fields(first[i])
            if all(got[f] == row[f] for f in fields):
                continue
        lines.append(f"Ticket {i + 1}: {row['ticket']}")
        lines.append(f"  {'correct answer':{width}s} : {row['label']}")
        for name, answers in columns:
            if i >= len(answers):
                lines.append(f"  {name:{width}s} : (waiting...)")
                continue
            text = " / ".join(l.strip() for l in answers[i].splitlines() if l.strip())
            if len(text) > 110:
                text = text[:107] + "..."
            got = extract_fields(answers[i])
            marks = "   ".join(f + (" \u2713" if got[f] == row[f] else " \u2717")
                               for f in fields)
            lines.append(f"  {name:{width}s} : {text or '(wrote nothing)'}")
            lines.append(f"  {'':{width}s}   {marks}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_card(columns, note=None):
    """Print one small results table: a column per run, a row per score.

    columns is a list of (name, scores) pairs. Pass RANDOM_GUESS as the scores
    for a random-guessing column.
    """
    rows = [("right format", "schema_valid"), ("category", "category_acc"),
            ("priority", "priority_acc"), ("queue", "queue_acc")]
    print("\n  " + f"{'':14s}" + "".join(f"{name:>13s}" for name, _ in columns))
    for label, key in rows:
        cells = "".join(f"{s[key]:13.0%}" if key in s else f"{'-':>13s}"
                        for _, s in columns)
        print(f"  {label:14s}{cells}")
    if note:
        print(f"\n  {note}")


def load_results():
    """Every run saved so far in out/results.json (empty if none yet)."""
    path = os.path.join(OUT_DIR, "results.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def print_scores(name, s):
    print(f"\n  === {name} ({s['n']} tickets) ===")
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
    return results


# --------------------------------------------------------------------------
# Small conveniences
# --------------------------------------------------------------------------

def param_report(model, label="model"):
    """Print how many of the model's numbers can be trained, out of the total."""
    def count(p):
        n = p.numel()
        # A 4-bit layer packs two numbers into each stored byte, so count both.
        if p.__class__.__name__ == "Params4bit":
            n *= 2 * getattr(getattr(p, "quant_storage", None), "itemsize", 1)
        return n
    total = sum(count(p) for p in model.parameters())
    trainable = sum(count(p) for p in model.parameters() if p.requires_grad)
    print(f"\n  {label}: {100 * trainable / total:.1f}% of its {total / 1e6:,.0f} "
          f"million numbers will be trained")
    return trainable, total


def quick_answer(model, tokenizer, ticket, max_new_tokens=24):
    """The model's answer to one ticket right now (first line only).
    Safe to call in the middle of training."""
    was_training = model.training
    model.eval()
    inputs = tokenizer(build_prompt(ticket), return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    if was_training:
        model.train()
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip().split("\n")[0].strip()


def loss_every(default):
    """How often to print the loss while training: every `default` steps, or
    the number given on the command line, as in:  --loss-every 6"""
    if "--loss-every" not in sys.argv:
        return default
    i = sys.argv.index("--loss-every")
    try:
        n = int(sys.argv[i + 1])
    except (IndexError, ValueError):
        n = 0
    if n < 1:
        raise SystemExit("  --loss-every needs a whole number of steps, "
                         "for example:  --loss-every 6")
    return n


# Check a mistyped --loss-every as soon as a lab starts, not after the model loads.
if "--loss-every" in sys.argv:
    loss_every(1)


def plain_progress(trainer, watch=None, answers=4):
    """Swap the Trainer's progress bar and raw log lines for one plain line
    every time it reports the loss (every logging_steps steps).

    watch, if given, is (tokenizer, row): `answers` times during training,
    also print the model's answer to that one test ticket, so you can see it
    learn.
    """
    from transformers import TrainerCallback
    from transformers.trainer_callback import PrinterCallback, ProgressCallback

    trainer.remove_callback(ProgressCallback)
    trainer.remove_callback(PrinterCallback)
    shown = {"next": 0, "every": 1, "last": 0}

    class PlainProgress(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            every = args.logging_steps
            what = ("each step" if every == 1 else
                    f"every {every} steps (the average over those {every} steps)")
            print(f"\n  Training, {state.max_steps} steps. Loss = how wrong its guesses "
                  f"are,\n  printed {what}. Watch it fall:", flush=True)
            if watch:
                shown["every"] = max(1, -(-state.max_steps // answers))   # rounded up
                shown["next"] = shown["every"]
                print(f"  Every {shown['every']} steps, its answer to test ticket 1, "
                      f"whose correct answer is\n    {watch[1]['label']}", flush=True)
                print("    before training", flush=True)
                show_answer(kwargs.get("model"))

        def on_log(self, args, state, control, logs=None, model=None, **kwargs):
            if logs and "loss" in logs:
                print(f"    step {state.global_step:>3} of {state.max_steps}   "
                      f"loss {float(logs['loss']):.3f}", flush=True)
                if watch and state.global_step >= shown["next"]:
                    show_answer(model, state.global_step)

        def on_train_end(self, args, state, control, model=None, **kwargs):
            # If the last step had no loss report, still show the final answer.
            if watch and shown["last"] < state.global_step:
                print(f"    after step {state.global_step}", flush=True)
                show_answer(model, state.global_step)

    def show_answer(model, step=0):
        if watch and model is not None:
            answer = quick_answer(model, watch[0], watch[1]["ticket"])
            if len(answer) > 66:
                answer = answer[:63] + "..."
            print(f"      answer: {answer or '(nothing)'}", flush=True)
            shown["last"] = step
            while shown["next"] <= step:
                shown["next"] += shown["every"]

    trainer.add_callback(PlainProgress())


def model_size_mb(model):
    """Approximate in-memory size of the model's parameters, in MB."""
    total = 0
    for p in model.parameters():
        # 4-bit numbers are stored two per byte, so each stored item is 1 byte
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total / 1024 ** 2


# --------------------------------------------------------------------------
# Optional: compare with a huge hosted model on Groq
# --------------------------------------------------------------------------

# Groq retires hosted models from time to time. If this step says the model
# is not found, pick another one (python scripts/check_groq.py lists them, and
# https://console.groq.com/docs/deprecations names the replacements) and change
# the two settings below. Nothing else in the workshop depends on them.
#
# That is also one of the arguments for fine-tuning a model you own: Lab 5's
# STEP 5 makes the point.
FRONTIER_MODEL = "openai/gpt-oss-120b"   # 120 billion numbers, about 5 billion
                                         # used for each word it writes

# This model always "thinks" before it answers; that can be turned down but not
# off. The settings below keep the thinking short and leave it out of the reply,
# so the reply is just the triage line. The thinking counts toward the reply's
# length limit, so the limit is well above what the triage line needs -- set it
# too low and the reply comes back empty.
FRONTIER_KWARGS = {
    "reasoning_effort": "low",
    "reasoning_format": "hidden",
    "max_completion_tokens": 256,
}


def groq_predictions(rows, model=FRONTIER_MODEL, show=0, **model_kwargs):
    """Answers from a large hosted model, prompt only (no training), for comparison.

    Returns None if the groq package is missing, GROQ_API_KEY is not set, or the
    request fails -- this step is optional and every lab must run without it.
    """
    if not os.environ.get("GROQ_API_KEY"):
        return None
    try:
        from groq import Groq
    except ImportError:
        print("  (groq package not installed -- pip install groq)")
        return None

    # When Groq answers "too many requests", its library waits and tries again
    # (twice by default). We allow more tries because a whole class sends
    # requests at once, and the free plan allows 30 per minute.
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
            # A model that doesn't "think" first rejects the two thinking
            # settings. Drop them and try once more, so switching models only
            # needs FRONTIER_MODEL changed.
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
