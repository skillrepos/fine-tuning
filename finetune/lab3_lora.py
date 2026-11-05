"""
Lab 3 - LoRA fine-tuning (with Hugging Face's PEFT library).

Same task as Lab 2, with a model four times bigger. This time we do NOT
retrain the whole model. LoRA keeps the original model frozen (unchanged) and
trains a small add-on next to it, called an adapter. Only the adapter learns,
and it is about 2% of the size of the model.

This lab also grades the model only on its answer (the triage line), not on
repeating the ticket text it was given. Lab 2 graded both.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments, DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

from common.triage import (
    load_tickets, build_prompt, generate_predictions,
    score, print_scores, save_result, param_report, OUT_DIR,
)

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
OUTPUT_DIR = os.path.join(OUT_DIR, "smollm2-lora")
MAX_STEPS = 24           # Lab 4 trains for the same number of steps
N_EVAL = 20


# ==========================================================================
# STEP 1 - Load the base model
# ==========================================================================
print("\n" + "=" * 70)
print(f"STEP 1: Load {BASE_MODEL}")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# Same fix as Lab 2: the padding token (filler that evens out lengths) must be
# different from the end token, or the model never learns where to stop.
# SmolLM2 already has a spare token we can use for padding.
if tokenizer.pad_token_id == tokenizer.eos_token_id:
    tokenizer.pad_token = "<|endoftext|>"
print(f"\n  pad id = {tokenizer.pad_token_id}, eos id = {tokenizer.eos_token_id}")

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
param_report(model, "SmolLM2-360M base (before LoRA)")


# ==========================================================================
# STEP 2 - Find out what the layers are actually called
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 2: Which layers can LoRA attach adapters to?")
print("=" * 70)

names = sorted({
    n.split(".")[-1]
    for n, m in model.named_modules()
    if isinstance(m, torch.nn.Linear)
})
print(f"\n  Layer names in this model: {names}")
print("""
  These are the layers LoRA can attach adapters to. The names depend on the
  model family, so a list copied from a different kind of model can match
  fewer layers than you meant -- or none. Always use the names the model
  itself reports. The percentage printed in Step 4 is your check.
""")


# ==========================================================================
# STEP 3 - Choose the LoRA settings
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 3: Configure LoRA")
print("=" * 70)

# TODO 1: set up LoRA -- adapter size (r) 16, strength (lora_alpha) 32,
#         dropout 0.05, no bias, a text-writing model (CAUSAL_LM), and the
#         seven layer names from Step 2. Leave use_dora=False for now.
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
)
print(f"\n  r (adapter size) = {peft_config.r}   lora_alpha (strength) = {peft_config.lora_alpha}")
print(f"  target_modules={peft_config.target_modules}")


# ==========================================================================
# STEP 4 - Add the adapters and see how much of the model will train
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 4: Add the LoRA adapters")
print("=" * 70)

# TODO 2: add the LoRA adapters to the model, then print how much of it
#         will be trained.
peft_model = None

print("""
  That percentage is how much of the model will actually be trained. The
  memory training needs grows with the part being trained, not with the whole
  model -- which is why LoRA fits where retraining everything would not.
""")


# ==========================================================================
# STEP 5 - Prepare the data, and grade only the answer
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 5: Prepare the training data (grade only the answer)")
print("=" * 70)

ds = load_tickets()


def encode(example):
    """Turn prompt + answer into tokens, and mark the prompt part 'do not grade'."""
    prompt = build_prompt(example["ticket"])
    full = prompt + example["label"] + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]

    # -100 means "do not grade this position". Everything before the answer is
    # the ticket we gave the model; we don't want it graded on repeating that.
    # TODO 3: build the labels (what the model is graded on): -100 for every
    #         prompt position, and the real token ids for the answer.
    labels = full_ids

    return {"input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels}


tokenized = ds.map(encode, remove_columns=ds["train"].column_names)

row = tokenized["train"][0]
n_masked = sum(1 for l in row["labels"] if l == -100)
print(f"\n  first example: {len(row['input_ids'])} tokens, "
      f"{n_masked} from the prompt (not graded), {len(row['labels']) - n_masked} graded")
print(f"  first 12 labels (-100 = not graded): {row['labels'][:12]}")
print("""
  In Lab 2 the model was also graded on repeating the ticket text, which wastes
  effort a small model can't spare. Here all of the training goes into writing
  the triage line.
""")


# ==========================================================================
# STEP 6 - Train
# ==========================================================================
print("\n" + "=" * 70)
print(f"STEP 6: Train the adapter, {MAX_STEPS} steps (about 3 minutes)")
print("=" * 70)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    max_steps=MAX_STEPS,        # a fixed number of steps keeps the lab on time
    per_device_train_batch_size=8,
    learning_rate=1e-3,         # step size; LoRA can use a bigger one than Lab 2
    warmup_steps=0.1,
    lr_scheduler_type="cosine",
    logging_steps=4,
    save_strategy="no",
    eval_strategy="no",
    report_to="none",
    use_cpu=True,
    optim="adamw_torch",
    seed=42,
)

# Pads every example in a batch to the same length, and marks the padding
# -100 so it is not graded either.
collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)

# TODO 4: build the Trainer from the model with adapters, the training data,
#         and the collator above.
trainer = Trainer(
    model=peft_model,
    args=args,
)

t0 = time.time()
result = trainer.train()
elapsed = time.time() - t0
print(f"\n  Trained in {elapsed:.0f}s   final loss {result.training_loss:.4f}")


# ==========================================================================
# STEP 7 - Save just the adapter
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 7: Save the adapter and compare artifact sizes")
print("=" * 70)

peft_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

adapter_mb = sum(
    os.path.getsize(os.path.join(OUTPUT_DIR, f))
    for f in os.listdir(OUTPUT_DIR)
    if os.path.isfile(os.path.join(OUTPUT_DIR, f))
) / 1024 ** 2
# Size of the model itself, without the adapters (4 bytes per number).
full_mb = sum(p.numel() * 4 for n, p in peft_model.get_base_model().named_parameters()
              if "lora_" not in n) / 1024 ** 2
print(f"\n  Files written: {sorted(os.listdir(OUTPUT_DIR))}")
print(f"  Adapter directory        : {adapter_mb:8.1f} MB")
print(f"  A full copy would be     : {full_mb:8.1f} MB")
print(f"  Ratio                    : {full_mb / adapter_mb:8.0f}x smaller")
print("""
  The model itself stays the same for every task or customer; each one only
  needs its own small adapter file. Twenty variants = one model + twenty
  small files, not twenty copies of the model.
""")


# ==========================================================================
# STEP 8 - Score the trained model
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 8: Evaluate the LoRA model")
print("=" * 70)

eval_rows = list(ds["eval"])[:N_EVAL]
preds = generate_predictions(peft_model, tokenizer, eval_rows, max_new_tokens=24, show=3)

s = score(preds, eval_rows)
print_scores("LoRA - SmolLM2-360M", s)
save_result("lora_smollm2", s, {
    "model": BASE_MODEL, "trained": True, "method": "lora",
    "rank": peft_config.r, "alpha": peft_config.lora_alpha, "steps": MAX_STEPS,
    "train_seconds": round(elapsed), "artifact_mb": round(adapter_mb, 1),
    "final_loss": round(result.training_loss, 4),
})

print("""
  Compare with Lab 2: a model four times bigger, only about 2% of it
  trained, and a saved file measured in megabytes instead of hundreds of
  megabytes.

  OPTIONAL: the end of Lab 3 in labs.md has two small experiments to try.
""")
