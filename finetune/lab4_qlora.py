"""
Lab 4 - QLoRA: 4-bit NormalFloat quantization + LoRA adapters, then export.

LoRA shrank the number of TRAINABLE parameters. It did nothing about the
frozen base model still sitting in memory at 2 or 4 bytes per weight.

QLoRA fixes that: quantize the frozen base to 4-bit NF4, keep the LoRA
adapters at higher precision, and backpropagate through the quantized weights.

This lab runs the EXACT recipe from Lab 3 -- same model, same data, same
rank, same steps -- with one thing changed: the base is 4-bit. That makes the
quality difference attributable to quantization and nothing else.

Then we merge the adapter back into a full-precision base and export a
standalone model.
"""

import sys, os, gc, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from common.triage import (
    load_tickets, to_prompt_completion, generate_predictions,
    score, print_scores, save_result, model_size_mb, OUT_DIR,
)

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"   # same model as Lab 3
BIG_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"     # Step 8: memory only
ADAPTER_DIR = os.path.join(OUT_DIR, "smollm2-qlora")
MERGED_DIR = os.path.join(OUT_DIR, "smollm2-merged")
MAX_STEPS = 24            # identical to Lab 3
N_EVAL = 20


# ==========================================================================
# STEP 1 - How big is this model at normal precision?
# ==========================================================================
print("\n" + "=" * 70)
print(f"STEP 1: Load {BASE_MODEL} at bfloat16 and measure it")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# Identical to Lab 3: point PAD at a spare existing token so it does not collide
# with EOS. We must NOT grow the vocabulary here -- you cannot cleanly resize the
# embedding table of an already-quantized model.
if tokenizer.pad_token_id == tokenizer.eos_token_id:
    tokenizer.pad_token = "<|endoftext|>"

bf16_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
n_params = sum(p.numel() for p in bf16_model.parameters())
bf16_mb = model_size_mb(bf16_model)
print(f"\n  parameters : {n_params:,}")
print(f"  bfloat16   : {bf16_mb:8,.0f} MB   (2 bytes per weight)")
print(f"  float32    : {bf16_mb * 2:8,.0f} MB   (4 bytes per weight -- what Lab 3 used)")

del bf16_model
gc.collect()


# ==========================================================================
# STEP 2 - Load the same model in 4-bit NF4
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 2: Load it again, quantized to 4-bit NormalFloat")
print("=" * 70)

# TODO 1: build the 4-bit config -- NF4 quant type, bfloat16 compute
#         dtype, double quantization on. Then load the model with it.
#         In transformers v5 you MUST pass quantization_config=;
#         load_in_4bit=True as a from_pretrained kwarg was removed.
bnb_config = None

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16,
    device_map={"": "cpu"},
)

nf4_mb = model_size_mb(model)
print(f"\n  4-bit NF4  : {nf4_mb:8,.0f} MB")
print(f"  vs bfloat16: {bf16_mb / nf4_mb:.1f}x smaller")
print(f"  vs float32 : {bf16_mb * 2 / nf4_mb:.1f}x smaller")
print("""
  Not quite the 8x you would predict from 32 bits -> 4 bits. Embeddings and
  the LM head stay unquantized, and every 64-weight block carries its own
  scale. In a small model those exceptions are a big fraction of the total;
  the bigger the model, the closer you get to the theoretical ratio.
""")


# ==========================================================================
# STEP 3 - Look at what actually changed inside the model
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 3: Inspect a quantized layer")
print("=" * 70)

layer = model.model.layers[0].self_attn.q_proj
print(f"\n  type       : {type(layer).__name__}")
print(f"  weight     : dtype={layer.weight.dtype}  shape={tuple(layer.weight.shape)}")
print(f"  quant_type : {layer.weight.quant_state.quant_type}")
print(f"  blocksize  : {layer.weight.quant_state.blocksize}")
print("""
  The weights are packed two-per-uint8, which is why the shape looks flattened.
  Quantization is BLOCKWISE: each block of 64 weights gets its own scale, so
  one outlier cannot wreck a whole tensor. NF4 spaces its 16 levels so they
  are information-theoretically optimal for normally-distributed values --
  which is what trained weights look like.

  Honest caveat for CPU: 4-bit buys you MEMORY, not speed. Every matmul has to
  dequantize first, so expect roughly 1.4x slower steps than bf16. On a GPU
  under VRAM pressure the memory saving is what lets you train at all -- and
  that is the whole point of QLoRA.
""")


# ==========================================================================
# STEP 4 - Prepare the quantized model for training, then add LoRA
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 4: Attach LoRA adapters to the quantized model")
print("=" * 70)

# TODO 2: prepare the quantized model for training (this casts layer
#         norms and the LM head back up to a stable dtype), then build
#         a LoRA config IDENTICAL to the one you used in Lab 3.
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
)
print("\n  Identical LoRA config to Lab 3. Only the base precision changed.")


# ==========================================================================
# STEP 5 - Train with the same recipe as Lab 3
# ==========================================================================
print("\n" + "=" * 70)
print(f"STEP 5: Train the QLoRA adapter, {MAX_STEPS} steps (about 3 minutes)")
print("=" * 70)

ds = load_tickets().map(to_prompt_completion)
train_ds = ds["train"].remove_columns(
    [c for c in ds["train"].column_names if c not in ("prompt", "completion")]
)

args = SFTConfig(
    output_dir=ADAPTER_DIR,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=8,
    learning_rate=1e-3,
    warmup_steps=0.1,
    lr_scheduler_type="cosine",
    logging_steps=4,
    save_strategy="no",
    eval_strategy="no",
    report_to="none",
    use_cpu=True,
    max_length=192,
    packing=False,
    optim="adamw_torch",
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    processing_class=tokenizer,
    peft_config=peft_config,
)
trainer.model.print_trainable_parameters()

t0 = time.time()
result = trainer.train()
elapsed = time.time() - t0
print(f"\n  Trained in {elapsed:.0f}s   final loss {result.training_loss:.4f}")

trainer.save_model(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
adapter_mb = sum(
    os.path.getsize(os.path.join(ADAPTER_DIR, f))
    for f in os.listdir(ADAPTER_DIR)
    if os.path.isfile(os.path.join(ADAPTER_DIR, f))
) / 1024 ** 2
print(f"  Adapter saved to {ADAPTER_DIR}  ({adapter_mb:.1f} MB)")

del model, trainer
gc.collect()


# ==========================================================================
# STEP 6 - Merge the adapter into a full-precision base and export
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 6: Merge the adapter and export a standalone model")
print("=" * 70)

print("\n  Reloading the base at bfloat16 (NOT 4-bit) to merge into...")
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
# TODO 3: attach the saved adapter to the bfloat16 base with
#         PeftModel.from_pretrained, then fold the adapter weights
#         into the base and drop the PEFT wrappers.
merged = base

print(f"  after merge_and_unload(): {type(merged).__name__}")
lora_left = [n for n, _ in merged.named_modules() if "lora" in n.lower()]
print(f"  LoRA modules remaining  : {len(lora_left)}")

merged.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)
merged_mb = sum(
    os.path.getsize(os.path.join(MERGED_DIR, f))
    for f in os.listdir(MERGED_DIR)
    if os.path.isfile(os.path.join(MERGED_DIR, f))
) / 1024 ** 2
print(f"  {MERGED_DIR}: {merged_mb:,.0f} MB  (vs {adapter_mb:.1f} MB for the adapter alone)")

print("""
  Read that step carefully -- it is the one people get wrong.

  We trained against a 4-BIT base but merged into a BFLOAT16 base. Merging
  into quantized weights would bake the quantization error permanently into
  the model. Train quantized, merge full-precision, then re-quantize the
  merged result if you need to.

  Merged means: no PEFT dependency at runtime, no adapter-loading step, one
  ordinary Hugging Face model directory. The trade is that you lose the
  swap-adapters-per-tenant trick from Lab 3. Decide per deployment.
""")


# ==========================================================================
# STEP 7 - Evaluate, and compare directly against Lab 3
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 7: Evaluate the merged QLoRA model")
print("=" * 70)

eval_rows = list(load_tickets()["eval"])[:N_EVAL]
preds = generate_predictions(merged, tokenizer, eval_rows, max_new_tokens=24, show=3)

s = score(preds, eval_rows)
print_scores("QLoRA - SmolLM2-360M (4-bit base)", s)
save_result("qlora_smollm2", s, {
    "model": BASE_MODEL, "trained": True, "method": "qlora",
    "rank": peft_config.r, "steps": MAX_STEPS,
    "train_seconds": round(elapsed),
    # Store the ADAPTER size so Lab 5's table compares like with like against
    # Lab 3. The merged directory is recorded separately.
    "artifact_mb": round(adapter_mb, 1), "merged_mb": round(merged_mb),
    "bf16_mb": round(bf16_mb), "nf4_mb": round(nf4_mb),
    "final_loss": round(result.training_loss, 4),
})

del merged, base
gc.collect()

print("""
  Now put this next to Lab 3. Same model, same data, same rank, same steps --
  the only difference is that the frozen base was 4-bit. Whatever accuracy gap
  you see IS the cost of quantization. In the QLoRA paper that gap is close to
  zero on real benchmarks; at this scale and step count expect a little noise
  in both directions.
""")


# ==========================================================================
# STEP 8 - Why this matters: the same trick on a 1.1B model
# ==========================================================================
print("\n" + "=" * 70)
print(f"STEP 8: Load {BIG_MODEL} in 4-bit (memory only, no training)")
print("=" * 70)

big = AutoModelForCausalLM.from_pretrained(
    BIG_MODEL, quantization_config=bnb_config,
    dtype=torch.bfloat16, device_map={"": "cpu"},
)
big_params = sum(p.numel() for p in big.parameters())
big_nf4_mb = model_size_mb(big)
print(f"\n  TinyLlama parameters : {big_params:,}")
print(f"  float32 would be     : {big_params * 4 / 1024 ** 2:8,.0f} MB")
print(f"  4-bit NF4 actual     : {big_nf4_mb:8,.0f} MB")
print(f"""
  That is the headline. A 1.1B model now occupies less memory than the 360M
  model did in Lab 3 at float32. Quantization is what moves the size of model
  you can fine-tune on given hardware up by roughly one tier.

  OPTIONAL, AT HOME: point BASE_MODEL at "microsoft/Phi-3-mini-4k-instruct"
  (3.8B) and rerun. Every other line stays the same. It is a 7.6 GB download
  and each step takes minutes on CPU, which is exactly why it is not a live
  lab -- but it is the same code, and that is the point.
""")
