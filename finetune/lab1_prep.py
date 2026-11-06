"""
Lab 1 - Dataset preparation and baseline evaluation.

Before you train anything, you need two things:
  1. The data, in the exact form the training code expects.
  2. A score to beat (the baseline): the best result you can get
     WITHOUT training, so you can prove later that training helped.

This script does both.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.triage import (
    load_tickets, to_prompt_completion, build_prompt,
    generate_predictions, score, print_scores, save_result,
    print_random_guess, pick_examples, best_prompt_predictions,
    CATEGORIES, PRIORITIES, QUEUES,
)

BASE_MODEL = "distilgpt2"
CHAT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"   # the model Labs 3 and 4 train
N_EVAL = 20          # keep the baseline quick -- 20 tickets is plenty to see the problem


# ==========================================================================
# STEP 1 - Load the ticket data
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 1: Load the ticket dataset")
print("=" * 70)

ds = load_tickets()
print(ds)
print("\nColumns:", ds["train"].column_names)
print("\nOne raw training row:")
for k, v in ds["train"][0].items():
    print(f"  {k:10s}: {v}")

print(f"\nLabel space: {len(CATEGORIES)} categories x {len(PRIORITIES)} priorities "
      f"x {len(QUEUES)} queues")


# ==========================================================================
# STEP 2 - Count how many tickets have each category and priority
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 2: Check the label distribution")
print("=" * 70)

from collections import Counter
for col in ["category", "priority"]:
    counts = Counter(ds["train"][col])
    print(f"\n  {col}:")
    for k, v in sorted(counts.items()):
        print(f"    {k:22s} {v:3d}  {'#' * v}")


# ==========================================================================
# STEP 3 - Split each row into a prompt and the answer we want back
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 3: Format as prompt / completion pairs")
print("=" * 70)

# TODO 1: run every row through to_prompt_completion() so each row gains
#         a "prompt" and a "completion" column.
formatted = None
row = formatted["train"][0]
print("\nPROMPT (what the model sees):")
print("-" * 60)
print(row["prompt"], end="")
print("-" * 60)
print("COMPLETION (what the model must produce):")
print("-" * 60)
print(row["completion"])
print("-" * 60)
print("\nThese two parts are not in the data file -- this step built them from each")
print("row. The prompt is the ticket inside a short fixed template; the completion")
print("is the row's label. Keeping them apart lets Lab 3 grade the model only on")
print("the completion, not on repeating the ticket back to us.")


# ==========================================================================
# STEP 4 - The same ticket in the format chat models expect
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 4: What a chat template does to the same example")
print("=" * 70)

chat_tok = AutoTokenizer.from_pretrained(CHAT_MODEL)
messages = [
    {"role": "system", "content": "You are a support ticket triage assistant."},
    {"role": "user", "content": build_prompt(ds["train"][0]["ticket"])},
]
# TODO 2: use the tokenizer to turn `messages` into chat-format text.
#         Ask for plain text (tokenize=False) that ends where the
#         model's reply should start (add_generation_prompt=True).
rendered = None
print(rendered)
print("-" * 60)
print("Same content, but wrapped in the special tokens THIS model family was")
print("trained on. Use the wrong template and quality quietly degrades.")


# ==========================================================================
# STEP 5 - Split the text into tokens and count them
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 5: How long are these sequences?")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token   # GPT-2 ships without a pad token

lengths = [
    len(tokenizer(r["prompt"] + r["completion"])["input_ids"])
    for r in formatted["train"]
]
lengths.sort()
print(f"\n  min {lengths[0]}  median {lengths[len(lengths) // 2]}  "
      f"p95 {lengths[int(len(lengths) * 0.95)]}  max {lengths[-1]} tokens")
print("\n  Short sequences are why this workshop fits on a CPU. Sequence length")
print("  drives cost quadratically in attention -- keep it tight.")


# ==========================================================================
# STEP 6 - Load the model before any training
# ==========================================================================
print("\n" + "=" * 70)
print(f"STEP 6: Load {BASE_MODEL} and look at it before any training")
print("=" * 70)

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
total = sum(p.numel() for p in model.parameters())
print(f"\n  {BASE_MODEL}: {total:,} parameters ({total / 1e6:.0f}M)")


# ==========================================================================
# STEP 7 - Get the untrained model's answers
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 7: Baseline predictions from the untrained model")
print("=" * 70)

eval_rows = list(ds["eval"])[:N_EVAL]
# TODO 3: get the model's answer for each test ticket. Show the first 3.
preds = None


# ==========================================================================
# STEP 8 - Score the baseline and save it
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 8: Score the baseline")
print("=" * 70)

# TODO 4: score the answers against the correct labels.
s = None
print_scores("BASELINE - distilgpt2, no training", s)
save_result("baseline_distilgpt2", s, {"model": BASE_MODEL, "trained": False})
print_random_guess()

print(f"""
  How these were measured: each of the {s['n']} answers is compared with its
  EXPECTED line.
    right format -- is the first line exactly in this form?
                    category: ... | priority: ... | queue: ...
    category / priority / queue correct -- does the answer name the right
                    one anywhere, in any form? ("Category: billing" on its
                    own line still counts.)

  Look at the PREDICTED lines above: distilgpt2 just keeps writing sentences
  and never names a category, priority or queue -- so it scores BELOW random
  guessing. That does not mean the model is hopeless. It means nobody told it
  what the task is. A bare prompt is not a fair starting point, so next we
  give a model the best prompt we can write.
""")


# ==========================================================================
# STEP 9 - The real score to beat: an untrained model with the best prompt
# ==========================================================================
print("\n" + "=" * 70)
print(f"STEP 9: Best prompt, no training -- {CHAT_MODEL.split('/')[-1]}")
print("=" * 70)

chat_model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL, dtype=torch.float32)
examples = pick_examples(ds["train"], k=len(CATEGORIES))
print("\n  Prompt = full instructions (every category, priority, queue and the")
print(f"  routing rules) + {len(examples)} worked examples, one ticket per category,")
print("  each with its correct answer.")

best_preds = best_prompt_predictions(chat_model, chat_tok, eval_rows, examples, show=3)
best = score(best_preds, eval_rows)
print_scores("BEST PROMPT - SmolLM2-360M-Instruct, no training", best)
save_result("bestprompt_smollm2", best, {"model": CHAT_MODEL, "trained": False,
                                         "method": "best prompt"})

print(f"""
  {'score':18s} {'random':>8s} {'bare prompt':>12s} {'best prompt':>12s}
  {'-' * 54}
  {'right format':18s} {'-':>8s} {s['schema_valid']:12.0%} {best['schema_valid']:12.0%}
  {'category correct':18s} {1/len(CATEGORIES):8.0%} {s['category_acc']:12.0%} {best['category_acc']:12.0%}
  {'priority correct':18s} {1/len(PRIORITIES):8.0%} {s['priority_acc']:12.0%} {best['priority_acc']:12.0%}
  {'queue correct':18s} {1/len(QUEUES):8.0%} {s['queue_acc']:12.0%} {best['queue_acc']:12.0%}

  The best-prompt column is the number every later lab has to beat. It is
  the best we can do without training -- and it costs a prompt of several
  hundred tokens on every single request. If training cannot clearly beat
  it, think hard about whether training is worth it.
""")
