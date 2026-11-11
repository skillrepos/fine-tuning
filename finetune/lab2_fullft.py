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


def tokenize(example):
    # TODO 1: build the training text: the prompt, then the correct answer,
    #         then the end-of-text token (tokenizer.eos_token) so the model
    #         learns where to stop.
    text = ""
    return tokenizer(text, truncation=True, max_length=192)


tokenized = ds.map(tokenize, remove_columns=ds["train"].column_names)

# Groups examples into batches and pads them to the same length.
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

# TODO 3: give the Trainer the training data, the collator and the tokenizer.
trainer = Trainer(
    model=model,
    args=args,
)

plain_progress(trainer)   # one plain line per loss report, no progress bar
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
eval_rows = list(ds["eval"])[:N_EVAL]
preds = generate_predictions(model, tokenizer, eval_rows, max_new_tokens=24)
s = score(preds, eval_rows)
save_result("fullft_distilgpt2", s, {
    "model": BASE_MODEL, "trained": True, "method": "full",
    "train_seconds": round(elapsed), "artifact_mb": round(size_mb),
    "steps": trainer.state.global_step,
})

print("\n" + "=" * 70)
print(f"RESULTS ({N_EVAL} test tickets)")
print("=" * 70)
bar = load_results().get("bestprompt_smollm2", {})
print_card([("best prompt", bar), ("Lab 2", s)],
           note="Score to beat: best prompt (Lab 1). Lab 2 trained a model 4x smaller.")
