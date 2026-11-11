"""
Lab 1 - The task, and the score to beat.

Before you train anything, find out the best result you can get WITHOUT
training. Every later lab has to beat it, or the training was not worth it.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common.quiet   # hide library warnings and progress bars

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.triage import (
    load_tickets, generate_predictions, score, save_result, print_card,
    pick_examples, best_prompt_predictions, best_prompt_messages,
    RANDOM_GUESS, CATEGORIES,
)

BASE_MODEL = "distilgpt2"                            # the model Lab 2 trains
CHAT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"   # the model Labs 3 and 4 train
N_EVAL = 20          # 20 test tickets keeps every lab quick


# ==========================================================================
# STEP 1 - The data
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 1: The data")
print("=" * 70)

ds = load_tickets()
print(f"\n  {len(ds['train'])} tickets to train on, {len(ds['eval'])} test tickets "
      "(never trained on). One of them:")
print(f"    ticket: {ds['train'][0]['ticket']}")
print(f"    answer: {ds['train'][0]['label']}")

eval_rows = list(ds["eval"])[:N_EVAL]


# ==========================================================================
# STEP 2 - An untrained model, given just the ticket
# ==========================================================================
print("\n" + "=" * 70)
print(f"STEP 2: {BASE_MODEL}, no training, just the ticket")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)

# TODO 1: get the model's answer to each test ticket. Show the first one.
preds = None
# TODO 2: score the answers against the correct ones.
s = None
save_result("baseline_distilgpt2", s, {"model": BASE_MODEL, "trained": False})


# ==========================================================================
# STEP 3 - An untrained chat model, given the best prompt we can write
# ==========================================================================
print("\n" + "=" * 70)
print(f"STEP 3: {CHAT_MODEL.split('/')[-1]}, no training, the best prompt we can write")
print("=" * 70)

chat_tok = AutoTokenizer.from_pretrained(CHAT_MODEL)
chat_model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL, dtype=torch.float32)
# The prompt: every label and rule, plus one worked example per category.
examples = pick_examples(ds["train"], k=len(CATEGORIES))
print(f"\n  The prompt: every label and rule, plus {len(examples)} worked examples.")
prompt_text = chat_tok.apply_chat_template(
    best_prompt_messages(eval_rows[0]["ticket"], examples),
    tokenize=False, add_generation_prompt=True)
prompt_tokens = len(chat_tok(prompt_text)["input_ids"])

best_preds = best_prompt_predictions(chat_model, chat_tok, eval_rows, examples)
best = score(best_preds, eval_rows)
save_result("bestprompt_smollm2", best, {"model": CHAT_MODEL, "trained": False,
                                         "method": "best prompt"})


# ==========================================================================
# Results
# ==========================================================================
print("\n" + "=" * 70)
print(f"RESULTS ({N_EVAL} test tickets)")
print("=" * 70)
print_card([("random", RANDOM_GUESS), ("just ticket", s), ("best prompt", best)],
           note=f"Best prompt = the score to beat. It costs a {prompt_tokens}-token "
                "prompt on every request.")
