"""
Lab 2 - Full fine-tuning: retrain every number in the model.

We use distilgpt2 (82 million numbers) because it is about the largest model
you can fully retrain on 4 CPU cores in a couple of minutes. Labs 3 and 4 use
cheaper methods and are compared with this one.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common.quiet   # hide library warnings and progress bars

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
)

from common.triage import (
    load_tickets, build_prompt, generate_predictions, score, save_result,
    load_results, print_card, param_report, plain_progress, OUT_DIR,
)

BASE_MODEL = "distilgpt2"
OUTPUT_DIR = os.path.join(OUT_DIR, "distilgpt2-fullft")
N_EVAL = 20


# ==========================================================================
# STEP 1 - Load the model
# ==========================================================================
print("\n" + "=" * 70)
print(f"STEP 1: Load {BASE_MODEL}")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# The model has no filler token for evening out lengths, so add one. It must
# be different from the end-of-text token: filler is never graded, and the
# model has to be graded on the end-of-text token to learn where to stop.
tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
model.resize_token_embeddings(len(tokenizer))
param_report(model, BASE_MODEL)


# ==========================================================================
# Turn each ticket and its answer into one training example
# ==========================================================================
ds = load_tickets()
eval_rows = list(ds["eval"])[:N_EVAL]


def tokenize(example):
    # TODO 1: build the training text: the prompt, then the correct answer,
    #         then the end-of-text marker (tokenizer.eos_token) so the model
    #         learns where to stop.
    text = ""
    return tokenizer(text, truncation=True, max_length=192)


tokenized = ds.map(tokenize, remove_columns=ds["train"].column_names)

# Write the first 3 training examples, exactly as the model trains on them.
EXAMPLES_FILE = os.path.join(OUT_DIR, "lab2_examples.txt")
with open(EXAMPLES_FILE, "w", encoding="utf-8") as f:
    f.write("The first 3 training examples, exactly as distilgpt2 trains on them:\n"
            "the prompt, then the correct answer, then the end-of-text marker.\n"
            "In this lab the model is graded on every word, the ticket included.\n")
    for i, row in enumerate(tokenized["train"].select(range(3))):
        f.write(f"\n----- Example {i + 1} ({len(row['input_ids'])} tokens, all graded) -----\n")
        f.write(tokenizer.decode(row["input_ids"]) + "\n")
print("\n  Wrote out/lab2_examples.txt: 3 training examples, as the model sees them"
      "\n  (see labs.md).")

# Groups the examples into batches of 8, adding filler so every example in
# a batch is the same length.
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# ==========================================================================
# STEP 2 - Train
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 2: Train (about 3 minutes)")
print("=" * 70)

# TODO 2: fill in the training settings below the output folder.
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
)

# TODO 3: give the Trainer the training examples, the batching helper above
#         (collator) and the tokenizer (which turns text into token numbers).
trainer = Trainer(
    model=model,
    args=args,
)

# One plain line per loss report, plus the model's answer to one test ticket
# at that moment, so you can watch it learn.
plain_progress(trainer, watch=(tokenizer, eval_rows[0]))
t0 = time.time()
trainer.train()
elapsed = time.time() - t0


# ==========================================================================
# Save the trained model
# ==========================================================================
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
size_mb = sum(
    os.path.getsize(os.path.join(OUTPUT_DIR, f))
    for f in os.listdir(OUTPUT_DIR)
    if os.path.isfile(os.path.join(OUTPUT_DIR, f))
) / 1024 ** 2
print(f"\n  Trained in {elapsed:.0f} seconds. Saved model: {size_mb:.0f} MB "
      "-- a whole new copy.")


# ==========================================================================
# Results
# ==========================================================================
preds = generate_predictions(model, tokenizer, eval_rows, max_new_tokens=24)
s = score(preds, eval_rows)
save_result("fullft_distilgpt2", s, {
    "model": BASE_MODEL, "trained": True, "method": "full",
    "train_seconds": round(elapsed), "artifact_mb": round(size_mb),
    "steps": trainer.state.global_step, "answers": preds,
})

print("\n" + "=" * 70)
print(f"RESULTS ({N_EVAL} test tickets)")
print("=" * 70)
bar = load_results().get("bestprompt_smollm2", {})
print_card([("best prompt", bar), ("Lab 2", s)],
           note="Score to beat: best prompt (Lab 1). Lab 2 trained a model 4x smaller.")
