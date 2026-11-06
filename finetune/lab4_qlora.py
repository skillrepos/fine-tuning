"""
Lab 4 - QLoRA: LoRA on a model stored in 4 bits, then export the result.

LoRA (Lab 3) cut down how much of the model trains. But the whole model still
has to sit in memory, at 2 or 4 bytes per number.

QLoRA shrinks that too: it stores the frozen (unchanging) model at 4 bits per
number and trains ordinary LoRA adapters on top of it.

READ THIS BEFORE YOU RUN IT. This lab is about how QLoRA works and how much
memory it saves, not about answer quality. We train for only 4 steps, and the
lab measures why: on a CPU, each 4-bit step is about 20-30 times slower than
Lab 3's, so a Lab 3-sized run would take over an hour. QLoRA is meant for
graphics cards (GPUs), where its memory savings pay off.

What you WILL come away with: the 4-bit memory numbers, proof that training
still works on top of a 4-bit model, the right way to fold the adapter in
and save one standalone model (a step many people get wrong), and a bigger
model loaded at 4 bits to show what the memory savings make possible.
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
    score, print_scores, save_result, model_size_mb, plain_progress, OUT_DIR,
)

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"   # same model as Lab 3
BIG_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"     # Step 8: memory only
ADAPTER_DIR = os.path.join(OUT_DIR, "smollm2-qlora")
MERGED_DIR = os.path.join(OUT_DIR, "smollm2-qlora-merged")
QLORA_STEPS = 4           # far too few to train fully -- see Step 5
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
#         Then load the model with them by passing quantization_config=
#         (the old shortcut, load_in_4bit=True on its own, no longer works).
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
print("""
  Not the full 8x you might expect from 32 bits down to 4. The parts that
  turn words into numbers and back stay at full size, and every group of 64
  numbers also stores one extra number to scale them. In a small model those
  extras are a big share of the total; the bigger the model, the closer you
  get to 8x.
""")


# ==========================================================================
# STEP 3 - Look at what actually changed inside the model
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 3: Look inside one 4-bit layer")
print("=" * 70)

layer = model.model.layers[0].self_attn.q_proj
print(f"\n  layer type        : {type(layer).__name__}")
print(f"  stored as         : {layer.weight.dtype} (whole bytes), shape {tuple(layer.weight.shape)}")
print(f"  4-bit format      : {layer.weight.quant_state.quant_type}")
print(f"  numbers per group : {layer.weight.quant_state.blocksize}")
print("""
  Two 4-bit numbers are packed into each byte, which is why the shape looks
  squashed into one column. The numbers are kept in groups of 64, each with
  its own scale, so one unusually large number can't throw off all the rest.
  The "nf4" format places its 16 possible values where a trained model's
  numbers actually tend to fall.

  The catch, which Step 5 will measure: 4 bits saves MEMORY but, on a CPU,
  costs a lot of SPEED. Before every calculation the numbers have to be
  unpacked back to 16 bits, and the 4-bit library has no fast way to do that
  on a CPU -- it was built for graphics cards (GPUs). On a GPU that is short
  of memory, this saving is what makes training possible at all. Here, on a
  CPU, it is a demonstration.
""")


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
print("\n  Same LoRA settings as Lab 3. The only change: the model underneath is 4-bit.")

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# ==========================================================================
# STEP 5 - Train 4 steps and time them
# ==========================================================================
print("\n" + "=" * 70)
print(f"STEP 5: Train {QLORA_STEPS} steps on top of the 4-bit model "
      f"(about 3 minutes)")
print("=" * 70)
print("""
  Four steps is not real training. It is a quick test that answers two
  questions:

    1. Can training still reach the adapters through a model stored at
       4 bits? (If not, QLoRA could not work.)
    2. How long does one 4-bit training step take on this machine,
       compared with Lab 3?

  We answer question 2 with a stopwatch.
""")

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
    warmup_steps=0,             # no easing in
    lr_scheduler_type="constant", # keep the learning rate the same throughout
    logging_steps=1,            # report the loss every step
    save_strategy="no",         # don't save copies during training
    eval_strategy="no",         # don't stop to test during training
    report_to="none",           # don't send results to outside tracking tools
    use_cpu=True,
    optim="adamw_torch",        # the standard method for applying each update
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
    print(f"\n  Each 4-bit step took {qlora_s_per_step / lora_s_per_step:.0f}x as long, "
          f"and handled a quarter as many examples.")
    print(f"  Per example, 4 bits is about "
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

print("""
  The loss changed, so training did reach the adapters through the 4-bit
  model. Question 1: yes, QLoRA works.

  Question 2 is the number to take back to your team. On a graphics card
  (GPU) the unpacking is fast, and the QLoRA paper reports it as roughly a
  third slower than ordinary 16-bit LoRA -- in exchange for fitting a
  65-billion-number model on one 48 GB card. On a CPU there is no fast way,
  so you pay the full cost.

  The rule: use QLoRA when memory is what stops you. If the model already
  fits, plain LoRA is faster.
""")

del model, trainer
gc.collect()


# ==========================================================================
# STEP 6 - Fold the adapter into a normal (not 4-bit) copy of the model, and save it
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 6: Fold the adapter into the model and save one standalone copy")
print("=" * 70)

print("\n  Reloading the model at 16 bits (NOT 4 bits) to fold the adapter into...")
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
# TODO 3: attach the saved adapter to the normal 16-bit model with
#         PeftModel.from_pretrained, then fold the adapter into the model
#         so it becomes one ordinary model again.
merged = base

print(f"  model type after folding in : {type(merged).__name__}")
lora_left = [n for n, _ in merged.named_modules() if "lora" in n.lower()]
print(f"  separate adapter pieces left: {len(lora_left)}")

merged.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)
merged_mb = sum(
    os.path.getsize(os.path.join(MERGED_DIR, f))
    for f in os.listdir(MERGED_DIR)
    if os.path.isfile(os.path.join(MERGED_DIR, f))
) / 1024 ** 2
print(f"  {MERGED_DIR}: {merged_mb:,.0f} MB  (vs {adapter_mb:.1f} MB for the adapter alone)")

print("""
  This is the step people most often get wrong.

  We trained on top of a 4-bit copy of the model, but folded the adapter into
  a 16-bit copy. Folding it into the 4-bit copy would lock the small rounding
  errors of 4-bit storage into the model for good. So: train at 4 bits, fold
  into a full-size copy, then shrink the result again if you need to.

  The folded-in model needs no extra library and no separate adapter file --
  it is one ordinary model folder. What you give up is Lab 3's trick of one
  shared model plus a small adapter per task or customer. Pick whichever
  suits how you will use it.

  These save steps are the same whether the adapter trained for 4 steps or
  4,000. What our 4-step adapter actually learned is another matter -- Step 7.
""")


# ==========================================================================
# STEP 7 - Score it: what 4 steps of training got us (not much)
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 7: Score the folded-in model -- and read it honestly")
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

print("""
  Those scores are poor, and they are supposed to be. The model saw only 8
  examples in total -- that is not fine-tuning. Don't read this as "QLoRA is
  worse than LoRA". Read it as: the number of training steps matters more
  than any other setting we have talked about today.

  A fair quality comparison -- same model, same data, same settings, 24 steps,
  32-bit vs 4-bit -- is worth doing. It is also about 70 minutes on a CPU, so
  it is homework, not a lab: set QLORA_STEPS = 24 and QLORA_BATCH = 8, run it
  overnight, and compare with Lab 3's row in out/results.json.

  If you want the expected answer first: the QLoRA paper found 4-bit training
  matched 16-bit LoRA quality on its tests. 4 bits trades speed for memory,
  not quality for memory.
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
# 4-bit numbers are packed two per byte, so simply counting what is stored
# finds only half of them. Each 4-bit layer also records its real count
# (in quant_state) -- use that.
big_params = sum(p.quant_state.shape.numel() if getattr(p, "quant_state", None) is not None
                 else p.numel() for p in big.parameters())
big_nf4_mb = model_size_mb(big)
print(f"\n  numbers in TinyLlama  : {big_params:,}")
print(f"  at 32 bits would be   : {big_params * 4 / 1024 ** 2:8,.0f} MB")
print(f"  at 4 bits, measured   : {big_nf4_mb:8,.0f} MB")
print(f"""
  This is the main takeaway, and it holds on any hardware. A model with 1.1
  billion numbers now takes less memory than the 360-million-number model did
  in Lab 3 at 32 bits. Storing at 4 bits lets the same machine fine-tune a
  model about three times bigger.

  On a GPU you would start training now. Here we stop: going by Step 5's
  timing, one 4-bit step of a model this size would take minutes on a CPU.

  OPTIONAL, AT HOME ON A GPU: set BASE_MODEL to TinyLlama and nothing else in
  this file has to change -- TinyLlama names its layers the same way SmolLM2
  does, so the target_modules list still matches. That is the real promise of
  QLoRA: not speed, but code that barely changes as the model grows.

  For "microsoft/Phi-3-mini-4k-instruct" (3.8 billion numbers) you need ONE
  more edit, because Phi-3 names its layers differently: target_modules
  becomes ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]. Keep the old
  names and the PEFT library stops with an error saying no layers matched --
  the same layer-name check as Lab 3 Step 2.
""")
