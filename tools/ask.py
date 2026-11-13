"""
Ask your trained model about tickets you write yourself.

Each ticket gets two answers:
  * before training: the original model, given the best prompt from Lab 1
  * after training : the same model with your Lab 3 adapter, given just the ticket

Under each answer: how sure that model is of each category.

Run it after Lab 3 (Lab 4 suggests when):
    python tools/ask.py                       type tickets, one per line
    python tools/ask.py "My laptop won't turn on"     answer one ticket and stop
"""

import sys, os, copy, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common.quiet   # hide library warnings and progress bars

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from common.triage import (
    load_tickets, build_prompt, pick_examples, CATEGORIES, ZERO_SHOT_SYSTEM, OUT_DIR,
)

ADAPTER_DIR = os.path.join(OUT_DIR, "smollm2-lora")   # what Lab 3 saved


def best_prompt(ticket, examples, tok):
    """Lab 1's best prompt: the rules, one worked example per category, then the ticket."""
    messages = [{"role": "system", "content": ZERO_SHOT_SYSTEM}]
    for ex in examples:
        messages.append({"role": "user", "content": ex["ticket"]})
        messages.append({"role": "assistant", "content": ex["label"]})
    messages.append({"role": "user", "content": ticket})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def answer(model, tok, prompt):
    """The model's one-line answer to a prompt."""
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40, do_sample=False,
                             eos_token_id=tok.eos_token_id,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip().split("\n")[0].strip()


def how_sure(model, tok, prompt):
    """How sure the model is of each category, right after it writes "category:".

    The model picks its next word piece from everything it knows. Here we look
    only at the first piece of each of the 8 category names, and share 100%
    out between them in proportion to how likely the model finds each one.
    Returns (category, share) pairs, the most likely first.
    """
    start = prompt + "category:"
    ids = {c: tok(start + " " + c)["input_ids"] for c in CATEGORIES}

    # Where the 8 names start to differ: everything before that is the same.
    first = ids[CATEGORIES[0]]
    n = min(next((i for i, (a, b) in enumerate(zip(first, v)) if a != b), len(v))
            for v in ids.values())
    with torch.no_grad():
        out = model(torch.tensor([first[:n]]), use_cache=True)
    after_start = torch.log_softmax(out.logits[0, -1], dim=-1)

    if len({v[n] for v in ids.values()}) == len(CATEGORIES):
        # Each name starts with its own piece: that one choice decides it.
        scores = {c: after_start[v[n]].item() for c, v in ids.items()}
    else:
        # Two names start the same way: score every piece of each name.
        scores = {}
        for c, v in ids.items():
            rest = v[n:]
            total = after_start[rest[0]].item()
            if len(rest) > 1:
                with torch.no_grad():
                    more = model(torch.tensor([rest[:-1]]),
                                 past_key_values=copy.deepcopy(out.past_key_values),
                                 use_cache=True)
                logp = torch.log_softmax(more.logits[0], dim=-1)
                total += sum(logp[i, t].item() for i, t in enumerate(rest[1:]))
            scores[c] = total

    top = max(scores.values())
    weights = {c: torch.tensor(s - top).exp().item() for c, s in scores.items()}
    whole = sum(weights.values())
    return sorted(((c, w / whole) for c, w in weights.items()), key=lambda x: -x[1])


def show(label, line, sure):
    print(f"\n  {label}")
    if len(line) > 74:                      # a model that rambles: show the start
        line = line[:71] + "..."
    print(f"    {line or '(wrote nothing)'}")
    print("    how sure of the category: "
          + "   ".join(f"{c} {s:.0%}" for c, s in sure[:3]))


# --------------------------------------------------------------------------
# Load the model and your adapter, once
# --------------------------------------------------------------------------
if not os.path.isdir(ADAPTER_DIR):
    raise SystemExit("\n  No Lab 3 adapter yet (out/smollm2-lora). Run Lab 3 first.")

with open(os.path.join(ADAPTER_DIR, "adapter_config.json")) as f:
    base_name = json.load(f)["base_model_name_or_path"]   # the model it fits

print("\n  Loading the model and your Lab 3 adapter...", flush=True)
tok = AutoTokenizer.from_pretrained(ADAPTER_DIR)   # saved with the adapter in Lab 3
model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.float32), ADAPTER_DIR)
model.eval()
examples = pick_examples(load_tickets()["train"], k=len(CATEGORIES))
print("  Ready.")


def triage(ticket):
    # Before training: switch the adapter off, and use Lab 1's best prompt.
    with model.disable_adapter():
        prompt = best_prompt(ticket, examples, tok)
        show("Before training (the original model, Lab 1's best prompt):",
             answer(model, tok, prompt), how_sure(model, tok, prompt))
    # After training: adapter on, and just the ticket, as it was trained.
    prompt = build_prompt(ticket)
    show("After training (your Lab 3 adapter, just the ticket):",
         answer(model, tok, prompt), how_sure(model, tok, prompt))


if len(sys.argv) > 1:
    triage(" ".join(sys.argv[1:]))
    raise SystemExit(0)

print("\n  Type a support ticket and press Enter. Press Enter on an empty line to stop.")
while True:
    try:
        ticket = input("\n  ticket> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not ticket:
        break
    triage(ticket)
