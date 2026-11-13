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
    load_tickets, build_prompt, pick_examples, extract_fields, right_format,
    save_result, print_card, write_answer_log,
    RANDOM_GUESS, CATEGORIES, ZERO_SHOT_SYSTEM, OUT_DIR,
)

BASE_MODEL = "distilgpt2"                            # the model Lab 2 trains
CHAT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"   # the model Labs 3 and 4 train
N_EVAL = 20          # 20 test tickets keeps every lab quick

PROMPT_FILE = os.path.join(OUT_DIR, "lab1_best_prompt.txt")
ANSWERS_FILE = os.path.join(OUT_DIR, "lab1_answers.txt")


# ==========================================================================
# How we ask a model a question
# ==========================================================================
def ask(model, tok, prompt, max_new_tokens):
    """Give a model some text, and return the text it writes next."""
    # TODO 1: turn the prompt into token numbers, let the model write up to
    #         max_new_tokens more, and turn just the new part back into text.
    return ""


# ==========================================================================
# How we build the best prompt
# ==========================================================================
def best_prompt(ticket, examples, tok):
    """The best prompt we can write: the rules, sample tickets with their correct answers, then the ticket."""
    # A chat model reads a conversation. "system" holds the instructions,
    # "user" is what a person wrote, "assistant" is the model's own reply.
    messages = [{"role": "system", "content": ZERO_SHOT_SYSTEM}]  # rules and every label
    # TODO 2: add each sample ticket as a "user" message followed by its
    #         correct answer as "assistant", then the new ticket as "user".
    # Join the conversation into the one long text the model actually reads.
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ==========================================================================
# How we score the answers
# ==========================================================================
def score_answers(answers, rows):
    """The share of answers in the right format, and with each field right."""
    right = {"format": 0, "category": 0, "priority": 0, "queue": 0}
    for answer, row in zip(answers, rows):
        right["format"] += right_format(answer)     # exactly "category: .. | priority: .. | queue: .."?
        # TODO 3: find the category, priority and queue the answer names, and
        #         count each one that matches the correct answer in row.
    n = len(rows)
    return {"n": n, "schema_valid": right["format"] / n,
            "category_acc": right["category"] / n,
            "priority_acc": right["priority"] / n, "queue_acc": right["queue"] / n}


# ==========================================================================
# STEP 1 - The data, and two files to read while the models work
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 1: The data, and two files to read while the models work")
print("=" * 70)

ds = load_tickets()
eval_rows = list(ds["eval"])[:N_EVAL]
print(f"\n  {len(ds['train'])} tickets to train on, {len(ds['eval'])} test tickets "
      "(never trained on).")

chat_tok = AutoTokenizer.from_pretrained(CHAT_MODEL)
examples = pick_examples(ds["train"], k=len(CATEGORIES))   # one per category
prompt_text = best_prompt(eval_rows[0]["ticket"], examples, chat_tok)
prompt_tokens = len(chat_tok(prompt_text)["input_ids"])
with open(PROMPT_FILE, "w", encoding="utf-8") as f:
    f.write(f"The exact text SmolLM2 reads for test ticket 1 in STEP 3 "
            f"({prompt_tokens} tokens).\n"
            "Each <|im_start|> line says who is speaking; the model writes its "
            "answer after the last one.\n"
            "For every other ticket, only the last ticket changes.\n\n"
            + "=" * 70 + "\n" + prompt_text)

plain, best = [], []            # answers from STEP 2 and STEP 3

def log():
    write_answer_log(ANSWERS_FILE, eval_rows,
                     [("distilgpt2 wrote", plain), ("SmolLM2 wrote", best)],
                     title="Each test ticket, its correct answer, and what each untrained "
                           "model wrote back.\n"
                           "distilgpt2 got only the ticket. SmolLM2 got the best prompt.")

log()
print("\n  Wrote two files to look at while the models work (see labs.md):")
print(f"    out/lab1_best_prompt.txt  the whole best prompt ({prompt_tokens} tokens)")
print("    out/lab1_answers.txt      every answer, filled in as it comes")


# ==========================================================================
# STEP 2 - An untrained model, given just the ticket
# ==========================================================================
print("\n" + "=" * 70)
print(f"STEP 2: {BASE_MODEL}, no training, just the ticket")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)

for i, row in enumerate(eval_rows):
    answer = ask(model, tokenizer, build_prompt(row["ticket"]), max_new_tokens=24)
    plain.append(answer.split("\n")[0].strip())     # keep only the first line
    log()
    print(f"\r  answered {i + 1} of {N_EVAL}", end="", flush=True)
print()

s = score_answers(plain, eval_rows)
save_result("baseline_distilgpt2", s, {"model": BASE_MODEL, "trained": False,
                                       "answers": plain})


# ==========================================================================
# STEP 3 - An untrained chat model, given the best prompt we can write
# ==========================================================================
print("\n" + "=" * 70)
print(f"STEP 3: {CHAT_MODEL.split('/')[-1]}, no training, the best prompt we can write")
print("=" * 70)

chat_model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL, dtype=torch.float32)

for i, row in enumerate(eval_rows):
    prompt = best_prompt(row["ticket"], examples, chat_tok)
    best.append(ask(chat_model, chat_tok, prompt, max_new_tokens=40))
    log()
    print(f"\r  answered {i + 1} of {N_EVAL}", end="", flush=True)
print()

best_s = score_answers(best, eval_rows)
save_result("bestprompt_smollm2", best_s, {"model": CHAT_MODEL, "trained": False,
                                           "method": "best prompt", "answers": best})


# ==========================================================================
# Results
# ==========================================================================
print("\n" + "=" * 70)
print(f"RESULTS ({N_EVAL} test tickets)")
print("=" * 70)
print_card([("random", RANDOM_GUESS), ("just ticket", s), ("best prompt", best_s)],
           note=f"Best prompt = the score to beat. It costs a {prompt_tokens}-token "
                "prompt on every request.")
