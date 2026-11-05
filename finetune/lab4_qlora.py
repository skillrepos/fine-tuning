"""
Lab 4 - QLoRA: LoRA on a model stored in 4 bits, then export the result.

LoRA (Lab 3) cut down how much of the model trains. But the whole model still
has to sit in memory, at 2 or 4 bytes per number.

QLoRA shrinks that too: it stores the frozen model in 4 bits per number (a
format called NF4) and trains normal-precision LoRA adapters on top of it.

READ THIS BEFORE YOU RUN IT. This lab is about how QLoRA works and how much
memory it saves, not about answer quality. We train for only 4 steps, and the
lab measures why: on a CPU, 4-bit math is roughly 20-25x slower per step than
the normal 32-bit math, so a Lab 3-sized run would take over an hour. QLoRA
is meant for graphics cards (GPUs), where its memory savings pay off.

What you WILL come away with: the 4-bit memory numbers, proof that training
still works through a 4-bit model, the right way to merge and export the
result (a step many people get wrong), and a bigger model loaded in 4 bits
to show what the memory savings make possible.
"""

import sys, os, gc, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    Trainer, TrainingArguments, DataCollatorForSeq2Seq,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from common.triage import (
    load_tickets, build_prompt, generate_predictions,
    score, print_scores, save_result, model_size_mb, OUT_DIR,
)

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"   # same model as Lab 3
BIG_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"     # Step 8: memory only
ADAPTER_DIR = os.path.join(OUT_DIR, "smollm2-qlora")
MERGED_DIR = os.path.join(OUT_DIR, "smollm2-qlora-merged")
QLORA_STEPS = 4           # far too few to train fully -- see Step 5
QLORA_BATCH = 2           # 4-bit math on a CPU slows down a lot with bigger batches
N_EVAL = 20


# ==========================================================================
# STEP 1 - How much memory does the model normally take?
# ==========================================================================
print("\n" + "=" * 70)
print(f"STEP 1: Load {BASE_MODEL} at bfloat16 and measure it")
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
print(f"\n  parameters : {n_params:,}")
print(f"  bfloat16   : {bf16_mb:8,.0f} MB   (2 bytes per weight)")
print(f"  float32    : {bf16_mb * 2:8,.0f} MB   (4 bytes per weight -- what Lab 3 used)")

del bf16_model
gc.collect()


# ==========================================================================
# STEP 2 - Load the same model again, stored in 4 bits per number
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 2: Load it again, quantized to 4-bit NormalFloat")
print("=" * 70)

# TODO 1: build the 4-bit settings -- the "nf4" format, math done in
#         bfloat16, and double quantization on. Then load the model with
#         them by passing quantization_config= (current versions no
#         longer accept load_in_4bit=True on its own).
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

  And here is the honest caveat, which Step 5 will make you measure yourself:
  4-bit buys MEMORY, and on a CPU it costs a great deal of SPEED. Every matmul
  has to dequantize a block before it can multiply, and bitsandbytes has no
  optimized CPU kernel for that path -- it is written for CUDA. On a GPU under
  VRAM pressure the memory saving is what lets the run happen at all, which is
  the entire point of QLoRA. On a CPU-only Codespace it is a demonstration.
""")


# ==========================================================================
# STEP 4 - Get the 4-bit model ready for training, then add LoRA
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 4: Attach LoRA adapters to the quantized model")
print("=" * 70)

# TODO 2: get the 4-bit model ready for training (this switches a few
#         sensitive layers back to higher precision so training stays
#         stable), then build LoRA settings IDENTICAL to Lab 3's.
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
)
print("\n  Identical LoRA config to Lab 3. Only the base precision changed.")

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# ==========================================================================
# STEP 5 - Train 4 steps and time them
# ==========================================================================
print("\n" + "=" * 70)
print(f"STEP 5: Train {QLORA_STEPS} steps through the quantized base "
      f"(about 3 minutes)")
print("=" * 70)
print("""
  Four steps is not training. It is a PROBE, and it answers two questions
  that matter more than a loss curve right now:

    1. Do gradients actually flow through 4-bit frozen weights to reach the
       adapters? (If the answer were no, QLoRA would not exist.)
    2. What does a 4-bit step cost on this hardware, versus Lab 3's float32?

  We will read the answer to #2 off the clock, not off a blog post.
""")

ds = load_tickets()


def encode(example):
    """Identical prompt masking to Lab 3."""
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
    warmup_steps=0,
    lr_scheduler_type="constant",
    logging_steps=1,
    save_strategy="no",
    eval_strategy="no",
    report_to="none",
    use_cpu=True,
    optim="adamw_torch",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True,
                                         label_pad_token_id=-100),
    processing_class=tokenizer,
)

t0 = time.time()
result = trainer.train()
elapsed = time.time() - t0
qlora_s_per_step = elapsed / QLORA_STEPS
print(f"\n  Trained {QLORA_STEPS} steps in {elapsed:.0f}s   "
      f"final loss {result.training_loss:.4f}")
print(f"  4-bit NF4, batch {QLORA_BATCH}: {qlora_s_per_step:.1f} s/step")

# Read Lab 3's training time back out of results.json so the comparison is
# between two runs YOU made on THIS machine, not two numbers from a slide.
results_path = os.path.join(OUT_DIR, "results.json")
lab3 = {}
if os.path.exists(results_path):
    with open(results_path) as f:
        lab3 = json.load(f).get("lora_smollm2", {})

if lab3.get("train_seconds") and lab3.get("steps"):
    lora_s_per_step = lab3["train_seconds"] / lab3["steps"]
    print(f"  Lab 3 float32, batch 8   : {lora_s_per_step:.1f} s/step")
    print(f"\n  Per step, 4-bit is {qlora_s_per_step / lora_s_per_step:.0f}x slower here "
          f"-- and it did 1/4 the examples per step.")
    print(f"  Per EXAMPLE, that is about "
          f"{(qlora_s_per_step / QLORA_BATCH) / (lora_s_per_step / 8):.0f}x.")
else:
    print("  (Run Lab 3 first to get the float32 comparison number.)")

model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
adapter_mb = sum(
    os.path.getsize(os.path.join(ADAPTER_DIR, f))
    for f in os.listdir(ADAPTER_DIR)
    if os.path.isfile(os.path.join(ADAPTER_DIR, f))
) / 1024 ** 2
print(f"\n  Adapter saved to {ADAPTER_DIR}  ({adapter_mb:.1f} MB)")

print("""
  The loss moved, so gradients reached the adapters through the packed 4-bit
  weights. Question 1 answered: QLoRA works.

  Question 2 answered too, and it is the number to take to your team. On CUDA,
  bitsandbytes dequantizes inside a fused kernel and the overhead is small --
  the QLoRA paper reports roughly a third slower than 16-bit LoRA, in exchange
  for fitting a 65B model on one 48 GB card. On CPU there is no such kernel,
  so you pay the dequantization in full.

  The rule that falls out of it: choose QLoRA when memory is the binding
  constraint. If the model already fits, plain LoRA is faster everywhere.
""")

del model, trainer
gc.collect()


# ==========================================================================
# STEP 6 - Fold the adapter into a normal (not 4-bit) copy of the model, and save it
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 6: Merge the adapter and export a standalone model")
print("=" * 70)

print("\n  Reloading the base at bfloat16 (NOT 4-bit) to merge into...")
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
# TODO 3: attach the saved adapter to the normal (bfloat16) model with
#         PeftModel.from_pretrained, then fold the adapter into the model
#         so it becomes one ordinary model again.
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

  This is the export procedure, and it is identical whether the adapter took
  4 steps or 4,000. What is inside our adapter after 4 steps is another
  matter -- Step 7.
""")


# ==========================================================================
# STEP 7 - Score it: what 4 steps of training got us (not much)
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 7: Evaluate the merged model -- and read it honestly")
print("=" * 70)

eval_rows = list(load_tickets()["eval"])[:N_EVAL]
preds = generate_predictions(merged, tokenizer, eval_rows, max_new_tokens=24, show=3)

s = score(preds, eval_rows)
print_scores(f"QLoRA probe - SmolLM2-360M, 4-bit base, {QLORA_STEPS} steps", s)
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

print("""
  Those scores are poor, and they are supposed to be. Eight examples of
  supervision is not fine-tuning. Do not read this row as "QLoRA is worse
  than LoRA" -- read it as "step count dominates every knob we have talked
  about all afternoon."

  The controlled quality comparison -- same model, same data, same rank, 24
  steps, float32 base vs 4-bit base -- is a real and worthwhile experiment.
  It is also a 70-minute CPU run, which is why it is a homework exercise and
  not a lab. Set QLORA_STEPS = 24 and QLORA_BATCH = 8, run it overnight, and
  compare against Lab 3's row in out/results.json.

  The published answer, if you want the expectation first: the QLoRA paper
  found 4-bit NF4 with double quantization matches 16-bit LoRA quality on
  their benchmarks. Quantization costs memory-for-speed, not memory-for-
  accuracy.
""")


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
# 4-bit numbers are packed two per byte, so a plain count (p.numel()) finds
# only half of them. quant_state remembers the real count -- use that.
big_params = sum(p.quant_state.shape.numel() if getattr(p, "quant_state", None) is not None
                 else p.numel() for p in big.parameters())
big_nf4_mb = model_size_mb(big)
print(f"\n  TinyLlama parameters : {big_params:,}")
print(f"  float32 would be     : {big_params * 4 / 1024 ** 2:8,.0f} MB")
print(f"  4-bit NF4 actual     : {big_nf4_mb:8,.0f} MB")
print(f"""
  That is the headline, and it is the one thing from this lab that transfers
  unchanged to your real hardware. A 1.1B model now occupies less memory than
  the 360M model did in Lab 3 at float32. Quantization moves the size of model
  you can fine-tune on given hardware up by roughly one tier.

  On a GPU you would now start training. Here we stop, because the arithmetic
  from Step 5 says a 1.1B 4-bit CPU step would take minutes, not seconds.

  OPTIONAL, AT HOME ON A GPU: point BASE_MODEL at TinyLlama and every other
  line in this file stays exactly the same -- TinyLlama is Llama-style, so the
  target_modules list already matches. That is the real claim of QLoRA: not
  that it is fast, but that the code barely changes as the model grows.

  If you go to "microsoft/Phi-3-mini-4k-instruct" (3.8B) instead, ONE more
  edit is needed: Phi-3 fuses its projections, so target_modules becomes
  ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]. Leave the Llama names
  in place and PEFT 0.21 will raise NoMatchingPeftModuleError -- which is the
  lesson from Lab 3 Step 2 arriving to collect.
""")
