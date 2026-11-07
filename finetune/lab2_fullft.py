"""
Lab 2 - Full fine-tuning.

Full fine-tuning updates every number (parameter) in the model. Labs 3 and 4
use cheaper methods, LoRA and QLoRA, and are compared with this one -- both
for how good the answers are and for what the training costs.

We use distilgpt2 (82 million parameters -- a smaller version of GPT-2)
because it is about the largest model you can fully fine-tune on 4 CPU cores
in a couple of minutes.
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
    load_tickets, build_prompt, generate_predictions,
    score, print_scores, save_result, param_report, plain_progress, OUT_DIR,
)

BASE_MODEL = "distilgpt2"
OUTPUT_DIR = os.path.join(OUT_DIR, "distilgpt2-fullft")
N_EVAL = 20


# ==========================================================================
# STEP 1 - Load the model and its tokenizer (turns text into tokens)
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 1: Load the model and its tokenizer (splits text into tokens)")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# GPT-2 has no padding token (filler that evens out lengths). Reusing the
# end token for padding is a common shortcut, but it breaks this lab -- the
# printout below explains why -- so we add a separate one.
tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
model.resize_token_embeddings(len(tokenizer))
param_report(model, "distilgpt2 (full fine-tuning)")

print(f"  padding token = {tokenizer.pad_token_id}    end-of-text token = {tokenizer.eos_token_id}   "
      "(two different tokens, on purpose)")
print("\n  -> Training will change 100% of these numbers. In Lab 3 it is about 2.3%.")


# ==========================================================================
# STEP 2 - Join the prompt and answer into one training example, as tokens
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 2: Build one training text per ticket, then split it into tokens")
print("=" * 70)

ds = load_tickets()


def tokenize(example):
    # TODO 1: build the full training text -- the prompt, then the correct
    #         answer, then the end-of-text token (tokenizer.eos_token) so the
    #         model learns where to stop. Then split it into tokens, keeping
    #         at most 192 (far more than any ticket needs).
    text = ""
    return tokenizer(text, truncation=True, max_length=192)


tokenized = ds.map(tokenize, remove_columns=ds["train"].column_names)
print(f"\n  {len(tokenized['train'])} training examples ready.")
print("\n  One training example, exactly as the model will see it:\n")
print(tokenizer.decode(tokenized["train"][0]["input_ids"]))

print("\n  -> It ends with <|endoftext|>, which teaches the model where to stop. Here")
print("     the model is graded on every word of it; Lab 3 grades only the triage line.")


# ==========================================================================
# STEP 3 - Group examples into batches and pad them to equal length
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 3: Group examples and pad them to the same length")
print("=" * 70)

# TODO 2: build the batching helper (collator) that pads each group of
#         examples to the same length. Use mlm=False (guess the next word).
collator = None
batch = collator([tokenized["train"][i] for i in range(4)])
# Pick the SHORTEST example in the group -- it is the one that got padded.
r = min(range(4), key=lambda i: len(tokenized["train"][i]["input_ids"]))
ids = batch["input_ids"][r].tolist()
graded = batch["labels"][r].tolist()
# Show the spot where the real text ends and the padding begins.
end = ids.index(tokenizer.pad_token_id) if tokenizer.pad_token_id in ids else len(ids)
start = max(0, end - 3)
print(f"\n  4 examples grouped together, each now {len(ids)} tokens long.")
print("  Where the shortest one ends and its padding begins:")
print(f"    tokens       : {ids[start:start + 8]}")
print(f"    what's graded: {graded[start:start + 8]}")
print(f"  -> -100 = skip when grading. Padding ({tokenizer.pad_token_id}) is skipped; "
      f"the end-of-text token ({tokenizer.eos_token_id}) is still graded.")


# ==========================================================================
# STEP 4 - Training settings
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 4: Choose the training settings")
print("=" * 70)

# TODO 3: fill in the training settings: 2 passes over the data (epochs),
#         8 examples per batch, learning rate 0.0003 (3e-4), slowly shrinking
#         (cosine), a loss report every 5 steps, no extra saved copies, and
#         CPU only.
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
)
print(f"\n  {args.num_train_epochs:.0f} passes through the data, "
      f"{args.per_device_train_batch_size} examples per step, "
      f"learning rate {args.learning_rate}")


# ==========================================================================
# STEP 5 - Train
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 5: Train (about 2-4 minutes on 4 CPU cores)")
print("=" * 70)

# TODO 4: build the Trainer: the model, the settings (args), the training
#         data, the collator, and the tokenizer -- passed as processing_class=.
trainer = Trainer(
    model=model,
    args=args,
)

plain_progress(trainer)   # one plain line per loss report, no progress bar
t0 = time.time()
result = trainer.train()
elapsed = time.time() - t0

print(f"\n  Trained in {elapsed:.0f}s")
print(f"  Average loss over the whole run: {result.training_loss:.4f}  (lower is better)")


# ==========================================================================
# STEP 6 - Save the model and check how much disk space it takes
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 6: Save the fine-tuned model")
print("=" * 70)

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

size_mb = sum(
    os.path.getsize(os.path.join(OUTPUT_DIR, f))
    for f in os.listdir(OUTPUT_DIR)
    if os.path.isfile(os.path.join(OUTPUT_DIR, f))
) / 1024 ** 2
print(f"\n  {OUTPUT_DIR} is {size_mb:.0f} MB")
print("  -> Retraining everything saves a whole new copy of the model. Compare with")
print("     the adapter size in Lab 3.")


# ==========================================================================
# STEP 7 - Score the trained model
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 7: Score the fine-tuned model")
print("=" * 70)

eval_rows = list(ds["eval"])[:N_EVAL]
preds = generate_predictions(model, tokenizer, eval_rows, max_new_tokens=24, show=3)

s = score(preds, eval_rows)
print_scores("FULL FINE-TUNE - distilgpt2", s)
save_result("fullft_distilgpt2", s, {
    "model": BASE_MODEL, "trained": True, "method": "full",
    "train_seconds": round(elapsed), "artifact_mb": round(size_mb),
    "steps": result.global_step, "final_loss": round(result.training_loss, 4),
})

print("\n  -> Compare with Lab 1's best-prompt row: category and queue should be clearly")
print("     higher. A small trained model beats a bigger prompted one.")
