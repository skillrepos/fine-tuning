"""
Lab 4 - QLoRA: LoRA on a model stored in 4 bits, then export the result.

LoRA (Lab 3) cut down how much of the model trains, but the whole model still
sits in memory at 2 or 4 bytes per number. QLoRA stores the frozen (unchanging)
model at 4 bits per number and trains ordinary LoRA adapters on top of it.

This lab is about how QLoRA works and how much memory it saves, not about
answer quality. It trains for only 4 steps: on a CPU each 4-bit step is about
20-30 times slower than Lab 3's. QLoRA saves memory, not time.
"""

import sys, os, gc, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common.quiet   # hide library warnings and progress bars

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    Trainer, TrainingArguments, DataCollatorForSeq2Seq,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from common.triage import (
    load_tickets, build_prompt, generate_predictions,
    score, print_scores, save_result, model_size_mb, plain_progress, param_report,
    OUT_DIR,
)

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"   # same model as Lab 3
BIG_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"     # STEP 8: memory only
ADAPTER_DIR = os.path.join(OUT_DIR, "smollm2-qlora")
MERGED_DIR = os.path.join(OUT_DIR, "smollm2-qlora-merged")
QLORA_STEPS = 4           # far too few to train fully -- see STEP 5
QLORA_BATCH = 2           # examples per step; 4-bit on a CPU slows down a lot with more
N_EVAL = 20


# ==========================================================================
# STEP 1 - How much memory does the model normally take?
# ==========================================================================
print("\n" + "=" * 70)
print(f"STEP 1: Load {BASE_MODEL} at its normal size and measure it")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# Same as Lab 3: use a spare existing token for padding, so it is different
# from the end token. We can't ADD a new token here (as Lab 2 did): a model
# already stored in 4 bits can't be cleanly resized to make room for one.
if tokenizer.pad_token_id == tokenizer.eos_token_id:
    tokenizer.pad_token = "<|endoftext|>"

bf16_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
n_params = sum(p.numel() for p in bf16_model.parameters())
bf16_mb = model_size_mb(bf16_model)
print(f"\n  numbers in the model     : {n_params:,}")
print(f"  at 16 bits (2 bytes each): {bf16_mb:8,.0f} MB")
print(f"  at 32 bits (4 bytes each): {bf16_mb * 2:8,.0f} MB   (what Lab 3 used)")

del bf16_model
gc.collect()


# ==========================================================================
# STEP 2 - Load the same model again, stored in 4 bits per number
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 2: Load it again, stored at 4 bits per number")
print("=" * 70)

# TODO 1: build the 4-bit settings -- the "nf4" format, math done in
#         16 bits (bfloat16), and the extra memory saver (double quant) on.
#         Then load the model, passing them as quantization_config=
#         so the model is stored in 4 bits.
bnb_config = None

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16,
    device_map={"": "cpu"},
)

nf4_mb = model_size_mb(model)
print(f"\n  at 4 bits (0.5 byte each): {nf4_mb:8,.0f} MB")
print(f"  That is {bf16_mb / nf4_mb:.1f}x smaller than 16 bits, "
      f"and {bf16_mb * 2 / nf4_mb:.1f}x smaller than 32 bits.")
print("  (Not 8x: some parts of the model stay at full size.)")


# ==========================================================================
# STEP 3 - Look at what actually changed inside the model
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 3: Look inside one 4-bit layer")
print("=" * 70)

layer = model.model.layers[0].self_attn.q_proj
print(f"\n  layer type        : {type(layer).__name__}  (a 4-bit layer)")
print(f"  stored as         : whole bytes, {layer.weight.shape[0]:,} rows x {layer.weight.shape[1]} column")
print(f"  4-bit format      : {layer.weight.quant_state.quant_type}")
print(f"  numbers per group : {layer.weight.quant_state.blocksize}")
print("\n  -> Two 4-bit numbers per byte, kept in groups of 64, each with its own scale.")


# ==========================================================================
# STEP 4 - Get the 4-bit model ready for training, then add LoRA
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 4: Add LoRA adapters to the 4-bit model")
print("=" * 70)

# TODO 2: get the 4-bit model ready for training (this switches a few
#         delicate layers to full size so training stays steady), then
#         build LoRA settings IDENTICAL to Lab 3's.
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
)
print("\n  Same LoRA settings as Lab 3; only the model underneath is 4-bit.")

model = get_peft_model(model, peft_config)
param_report(model, "SmolLM2-360M, 4-bit, with LoRA adapters")


# ==========================================================================
# STEP 5 - Train 4 steps and time them
# ==========================================================================
print("\n" + "=" * 70)
print(f"STEP 5: Train {QLORA_STEPS} steps on top of the 4-bit model "
      f"(about 3 minutes)")
print("=" * 70)
print(f"\n  {QLORA_STEPS} steps is a quick test, not real training.")

ds = load_tickets()


def encode(example):
    """Same as Lab 3: grade only the answer, not the ticket."""
    prompt = build_prompt(example["ticket"])
    full = prompt + example["label"] + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels}


tokenized = ds.map(encode, remove_columns=ds["train"].column_names)

args = TrainingArguments(
    output_dir=ADAPTER_DIR,
    max_steps=QLORA_STEPS,
    per_device_train_batch_size=QLORA_BATCH,
    learning_rate=1e-3,
    lr_scheduler_type="constant", # keep the learning rate the same throughout
    logging_steps=1,            # report the loss every step
    save_strategy="no",         # don't save extra copies of the model while training
    use_cpu=True,               # run on the processor, not a graphics card
    optim="adamw_torch",        # the update rule: AdamW, a version of Adam (see slides)
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True,
                                         label_pad_token_id=-100),
    processing_class=tokenizer,
)

plain_progress(trainer)   # one plain line per loss report, no progress bar
t0 = time.time()
result = trainer.train()
elapsed = time.time() - t0
qlora_s_per_step = elapsed / QLORA_STEPS
print(f"\n  Trained {QLORA_STEPS} steps in {elapsed:.0f}s   "
      f"average loss {result.training_loss:.4f}")
print(f"  4-bit, {QLORA_BATCH} examples per step : {qlora_s_per_step:5.1f} seconds per step")

# Read Lab 3's training time back out of results.json so the comparison is
# between two runs YOU made on THIS machine, not two numbers from a slide.
results_path = os.path.join(OUT_DIR, "results.json")
lab3 = {}
if os.path.exists(results_path):
    with open(results_path) as f:
        lab3 = json.load(f).get("lora_smollm2", {})

if lab3.get("train_seconds") and lab3.get("steps"):
    lora_s_per_step = lab3["train_seconds"] / lab3["steps"]
    print(f"  Lab 3, 8 examples per step : {lora_s_per_step:5.1f} seconds per step")
    print(f"  -> per example, 4 bits is about "
          f"{(qlora_s_per_step / QLORA_BATCH) / (lora_s_per_step / 8):.0f}x slower here.")
else:
    print("  (Run Lab 3 first to get its timing to compare with.)")

model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
adapter_mb = sum(
    os.path.getsize(os.path.join(ADAPTER_DIR, f))
    for f in os.listdir(ADAPTER_DIR)
    if os.path.isfile(os.path.join(ADAPTER_DIR, f))
) / 1024 ** 2
print(f"\n  Adapter saved to {ADAPTER_DIR}  ({adapter_mb:.1f} MB)")

print("\n  -> The loss fell, so training works at 4 bits -- just slowly on a CPU.")
print("     QLoRA saves memory, not time.")

del model, trainer
gc.collect()


# ==========================================================================
# STEP 6 - Fold the adapter into a normal (not 4-bit) copy of the model, and save it
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 6: Fold the adapter into the model and save one standalone copy")
print("=" * 70)

print("\n  Reloading at 16 bits (NOT 4 bits)...")
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
# TODO 3: attach the saved adapter to the normal 16-bit model with
#         PeftModel.from_pretrained, then fold the adapter into the model
#         so it becomes one ordinary model again.
merged = base

print(f"  still needs the adapter library : {'yes' if type(merged).__name__.startswith('Peft') else 'no'}")
lora_left = [n for n, _ in merged.named_modules() if "lora" in n.lower()]
print(f"  separate adapter pieces left    : {len(lora_left)}")

merged.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)
merged_mb = sum(
    os.path.getsize(os.path.join(MERGED_DIR, f))
    for f in os.listdir(MERGED_DIR)
    if os.path.isfile(os.path.join(MERGED_DIR, f))
) / 1024 ** 2
print(f"  {MERGED_DIR}: {merged_mb:,.0f} MB  (vs {adapter_mb:.1f} MB for the adapter alone)")

print("\n  -> Folded into a 16-bit copy, not the 4-bit one, so its rounding errors")
print("     are not locked in. The result is one ordinary model folder.")


# ==========================================================================
# STEP 7 - Score it: what 4 steps of training got us (not much)
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 7: Score the folded-in model")
print("=" * 70)

eval_rows = list(load_tickets()["eval"])[:N_EVAL]
preds = generate_predictions(merged, tokenizer, eval_rows, max_new_tokens=24, show=3)

s = score(preds, eval_rows)
print_scores(f"QLoRA test run - SmolLM2-360M, 4-bit, {QLORA_STEPS} steps", s)
save_result("qlora_smollm2", s, {
    "model": BASE_MODEL, "trained": True, "method": "qlora (4-step probe)",
    "rank": peft_config.r, "steps": QLORA_STEPS, "batch_size": QLORA_BATCH,
    "train_seconds": round(elapsed),
    "sec_per_step": round(qlora_s_per_step, 1),
    # Save the ADAPTER size, so Lab 5's table compares it fairly with Lab 3's
    # adapter. The merged model's size is saved separately.
    "artifact_mb": round(adapter_mb, 1), "merged_mb": round(merged_mb),
    "bf16_mb": round(bf16_mb), "nf4_mb": round(nf4_mb),
    "final_loss": round(result.training_loss, 4),
    "note": "mechanics probe, not a quality run",
})

del merged, base
gc.collect()

print(f"\n  -> Poor on purpose: {QLORA_STEPS * QLORA_BATCH} examples is not training.")
print("     The fair 24-step comparison is homework (end of Lab 4 in labs.md).")


# ==========================================================================
# STEP 8 - Why it matters: a 1.1-billion-parameter model in 4 bits
# ==========================================================================
print("\n" + "=" * 70)
print(f"STEP 8: Load {BIG_MODEL} in 4-bit (memory only, no training)")
print("=" * 70)

big = AutoModelForCausalLM.from_pretrained(
    BIG_MODEL, quantization_config=bnb_config,
    dtype=torch.bfloat16, device_map={"": "cpu"},
)
# 4-bit numbers are packed two per byte, so simply counting what is stored
# finds only half of them. Each 4-bit layer also records its real count
# (in quant_state) -- use that.
big_params = sum(p.quant_state.shape.numel() if getattr(p, "quant_state", None) is not None
                 else p.numel() for p in big.parameters())
big_nf4_mb = model_size_mb(big)
print(f"\n  numbers in TinyLlama  : {big_params:,}")
print(f"  at 32 bits would be   : {big_params * 4 / 1024 ** 2:8,.0f} MB")
print(f"  at 4 bits, measured   : {big_nf4_mb:8,.0f} MB")
print("\n  -> A model 3x bigger now fits in less memory than Lab 3's did at 32 bits.")
print("     On a GPU you would train it now.")
