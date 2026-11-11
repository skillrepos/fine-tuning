"""
Lab 4 - QLoRA: LoRA on a model stored in 4 bits, then fold it into one model.

LoRA (Lab 3) trains only a small adapter, but the whole model still sits in
memory at 4 bytes per number. QLoRA stores the unchanged model at half a byte
per number and trains an ordinary LoRA adapter on top. Then we fold the
adapter into the model, so there is one ordinary model to ship.

This lab shows how it works and what it saves. It trains for only a few steps:
on a CPU, 4-bit training is slow. QLoRA saves memory, not time.
"""

import sys, os, gc, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common.quiet   # hide library warnings and progress bars

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    Trainer, TrainingArguments, DataCollatorForSeq2Seq,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from common.triage import (
    load_tickets, build_prompt, save_result, load_results, model_size_mb,
    plain_progress, OUT_DIR,
)

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"   # same model as Lab 3
ADAPTER_DIR = os.path.join(OUT_DIR, "smollm2-qlora")
MERGED_DIR = os.path.join(OUT_DIR, "smollm2-qlora-merged")
QLORA_STEPS = 3           # enough to see the loss fall; far too few to train fully
QLORA_BATCH = 2           # examples per step; 4 bits on a CPU slows down a lot with more


def folder_mb(path):
    return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)
               if os.path.isfile(os.path.join(path, f))) / 1024 ** 2


# ==========================================================================
# STEP 1 - Load the model stored in 4 bits, and measure the memory saved
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 1: Load the Lab 3 model, stored in 4 bits")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# Same filler token as Lab 3 (different from the end-of-text token).
if tokenizer.pad_token_id == tokenizer.eos_token_id:
    tokenizer.pad_token = "<|endoftext|>"

# TODO 1: build the 4-bit settings -- the "nf4" format, math done in
#         16 bits (bfloat16), and the extra memory saver (double quant) on.
#         Then pass them to the loader below as quantization_config=.
bnb_config = None

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16,
    device_map={"": "cpu"},
)

# 4-bit numbers are packed two per byte; each 4-bit layer records its real count.
n_params = sum(p.quant_state.shape.numel() if getattr(p, "quant_state", None) is not None
               else p.numel() for p in model.parameters())
fp32_mb = n_params * 4 / 1024 ** 2
nf4_mb = model_size_mb(model)
big_mb = nf4_mb * 1.1e9 / n_params      # a 1.1-billion-number model, same storage

print(f"\n  in memory at 32 bits (Lab 3): {fp32_mb:7,.0f} MB")
print(f"  in memory at 4 bits         : {nf4_mb:7,.0f} MB   ({fp32_mb / nf4_mb:.1f}x smaller)")
print(f"  -> A model 3x bigger (1.1 billion numbers) would need only about "
      f"{big_mb:,.0f} MB at 4 bits.")


# ==========================================================================
# STEP 2 - Add LoRA adapters and train a few steps
# ==========================================================================
print("\n" + "=" * 70)
print(f"STEP 2: Train a LoRA adapter on top of the 4-bit model ({QLORA_STEPS} steps)")
print("=" * 70)

# Get the 4-bit model ready to train. (Skip a memory-saving option that
# would make training even slower.)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
model = get_peft_model(model, LoraConfig(        # same settings as Lab 3
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
))

ds = load_tickets()


def encode(example):
    """Same as Lab 3: grade only the answer, not the ticket."""
    prompt = build_prompt(example["ticket"])
    full = prompt + example["label"] + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids),
            "labels": labels}


tokenized = ds.map(encode, remove_columns=ds["train"].column_names)

args = TrainingArguments(
    output_dir=ADAPTER_DIR,
    max_steps=QLORA_STEPS,
    per_device_train_batch_size=QLORA_BATCH,
    learning_rate=1e-3,
    lr_scheduler_type="constant",   # the same size of adjustment every step
    logging_steps=1,                # report the loss every step
    save_strategy="no",
    use_cpu=True,
    optim="adamw_torch",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    processing_class=tokenizer,
)

plain_progress(trainer)   # one plain line per loss report, no progress bar
t0 = time.time()
trainer.train()
elapsed = time.time() - t0

model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)

# Compare the time per training example with Lab 3's run on this machine.
s_per_example = elapsed / (QLORA_STEPS * QLORA_BATCH)
lab3 = load_results().get("lora_smollm2", {})
if lab3.get("train_seconds") and lab3.get("steps"):
    lab3_per_example = lab3["train_seconds"] / (lab3["steps"] * 8)
    print(f"\n  Seconds per training example: {s_per_example:.1f} at 4 bits, "
          f"{lab3_per_example:.1f} in Lab 3 ({s_per_example / lab3_per_example:.0f}x slower).")
else:
    print(f"\n  Seconds per training example: {s_per_example:.1f} at 4 bits.")

del model, trainer
gc.collect()


# ==========================================================================
# STEP 3 - Fold the adapter into a normal copy of the model
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 3: Fold the adapter into the model and save one model")
print("=" * 70)

# Fold into a 16-bit copy, not the 4-bit one, so the 4-bit rounding is not
# made permanent.
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
# TODO 2: attach the saved adapter to the 16-bit model (PeftModel.from_pretrained),
#         then fold it in so it becomes one ordinary model.
merged = base

merged.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)
merged_mb = folder_mb(MERGED_DIR)
print(f"\n  Saved one ordinary model: {merged_mb:,.0f} MB. "
      "It runs without the adapter file or the LoRA library.")

save_result("qlora_smollm2", {}, {
    "model": BASE_MODEL, "method": "qlora (few-step test)",
    "steps": QLORA_STEPS, "batch_size": QLORA_BATCH, "train_seconds": round(elapsed),
    "artifact_mb": round(folder_mb(ADAPTER_DIR), 1), "merged_mb": round(merged_mb),
    "fp32_mb": round(fp32_mb), "nf4_mb": round(nf4_mb),
})
