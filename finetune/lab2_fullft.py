"""
Lab 2 - Full fine-tuning with the Hugging Face Trainer.

Every parameter in the model gets an update. This is the baseline that LoRA
and QLoRA are measured against -- both in quality and in cost.

We use distilgpt2 (82M parameters -- it is the 6-layer distillation of GPT-2's
124M) because it is the largest model you can honestly full-fine-tune on 4 CPU
cores in a couple of minutes.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
)

from common.triage import (
    load_tickets, build_prompt, generate_predictions,
    score, print_scores, save_result, param_report, OUT_DIR,
)

BASE_MODEL = "distilgpt2"
OUTPUT_DIR = os.path.join(OUT_DIR, "distilgpt2-fullft")
N_EVAL = 20


# ==========================================================================
# STEP 1 - Model and tokenizer
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 1: Load model and tokenizer")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# GPT-2 ships without a padding token. The reflex is `pad_token = eos_token`,
# and it quietly breaks this lab -- see the note below.
tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
model.resize_token_embeddings(len(tokenizer))
param_report(model, "distilgpt2 (full fine-tuning)")

print(f"""
  100% trainable. Hold on to that number -- in Lab 3 it becomes about 2.3%.

  Why a dedicated <|pad|> token instead of reusing EOS?
  The collator in Step 3 sets every PAD position in the labels to -100 so
  padding does not contribute to the loss. If PAD and EOS are the same token
  id, that also erases every real EOS from the labels -- the model never
  learns where to stop, and generation runs on forever. Two distinct tokens,
  one bug avoided.
    pad id = {tokenizer.pad_token_id}, eos id = {tokenizer.eos_token_id}
""")


# ==========================================================================
# STEP 2 - Tokenize prompt + completion into one training sequence
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 2: Tokenize the dataset")
print("=" * 70)

ds = load_tickets()


def tokenize(example):
    # TODO 1: build the full training string -- prompt, then the gold label,
    #         then an EOS token so the model learns where to stop.
    #         Then tokenize it with truncation at 192 tokens.
    text = ""
    return tokenizer(text, truncation=True, max_length=192)


tokenized = ds.map(tokenize, remove_columns=ds["train"].column_names)
print(tokenized)
print(f"\n  First example decoded back:\n")
print(tokenizer.decode(tokenized["train"][0]["input_ids"]))

print("""
  Note what we did NOT do: mask the prompt. In plain causal-LM fine-tuning the
  model is graded on predicting every token, ticket text included. Lab 3 masks
  the prompt out of the loss -- watch for the difference.
""")


# ==========================================================================
# STEP 3 - The collator: dynamic padding, causal (not masked) LM
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 3: Build the data collator")
print("=" * 70)

# TODO 2: build a causal-LM collator (mlm=False), not a masked-LM one.
collator = None
batch = collator([tokenized["train"][i] for i in range(4)])
# Pick the SHORTEST row in the batch -- that is the one that actually got
# padded, so it is the one where you can see -100 appear.
r = min(range(4), key=lambda i: len(tokenized["train"][i]["input_ids"]))
print(f"\n  batch input_ids : {tuple(batch['input_ids'].shape)}")
print(f"  batch labels    : {tuple(batch['labels'].shape)}")
print(f"  padded positions in labels are set to -100 (ignored by the loss):")
print(f"  input_ids tail : {batch['input_ids'][r][-8:].tolist()}")
print(f"  labels tail    : {batch['labels'][r][-8:].tolist()}")
print(f"\n  The EOS token (id {tokenizer.eos_token_id}) survives in the labels -- that is")
print(f"  what teaches the model to stop. Had we aliased PAD to EOS, every one")
print(f"  of those would read -100 instead.")
print("\n  mlm=False means causal language modelling -- predict the NEXT token.")


# ==========================================================================
# STEP 4 - Training arguments
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 4: Configure training")
print("=" * 70)

# TODO 3: fill in the training arguments. 2 epochs, batch size 8,
#         lr 3e-4, cosine schedule, log every 5 steps, no saving,
#         no reporting, and force CPU.
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
)
print(f"\n  epochs={args.num_train_epochs}  batch={args.per_device_train_batch_size}  "
      f"lr={args.learning_rate}")


# ==========================================================================
# STEP 5 - Train
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 5: Train (about 2-4 minutes on 4 CPU cores)")
print("=" * 70)

# TODO 4: assemble the Trainer. Careful: in transformers v5 the tokenizer
#         argument is called processing_class.
trainer = Trainer(
    model=model,
    args=args,
)

t0 = time.time()
result = trainer.train()
elapsed = time.time() - t0

print(f"\n  Trained in {elapsed:.0f}s")
print(f"  Final training loss: {result.training_loss:.4f}")


# ==========================================================================
# STEP 6 - Save the model and measure what that cost on disk
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
print("""
  A full fine-tune produces a whole new copy of the model. Ten fine-tuned
  variants means ten copies. Remember this number when you see the adapter
  size in Lab 3.
""")


# ==========================================================================
# STEP 7 - Evaluate
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 7: Evaluate the fine-tuned model")
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

print("""
  Compare against the Lab 1 baseline.

  Expect schema validity to jump dramatically -- the model learned the SHAPE
  of the answer almost immediately. Category accuracy will be much weaker.
  82M parameters can learn a format; learning to actually reason about which
  category a ticket belongs to needs a bigger model. That is Lab 3.
""")
