"""
Lab 3 - LoRA: train a small add-on instead of the whole model.

Same task as Lab 2, with a model four times bigger. LoRA leaves the model
itself unchanged and trains a small add-on beside it, called an adapter --
about 2% of the model's size. This lab also grades the model only on its
answer, not on repeating the ticket it was given.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common.quiet   # hide library warnings and progress bars

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments, DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

from common.triage import (
    load_tickets, build_prompt, generate_predictions, score, save_result,
    load_results, print_card, param_report, plain_progress, write_answer_log,
    OUT_DIR,
)

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
OUTPUT_DIR = os.path.join(OUT_DIR, "smollm2-lora")
MAX_STEPS = 24
N_EVAL = 20


# ==========================================================================
# STEP 1 - Load the model and add the adapters
# ==========================================================================
print("\n" + "=" * 70)
print(f"STEP 1: Load {BASE_MODEL.split('/')[-1]} and add LoRA adapters")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# As in Lab 2, the filler token must differ from the end-of-text token.
# This model already has a spare token to use as filler.
if tokenizer.pad_token_id == tokenizer.eos_token_id:
    tokenizer.pad_token = "<|endoftext|>"

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)

# TODO 1: add the rest of the LoRA settings, including which layers get adapters.
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
)

# TODO 2: add the adapters to the model, then report how much will be trained.
peft_model = None


# ==========================================================================
# Prepare the data, and grade only the answer
# ==========================================================================
ds = load_tickets()
eval_rows = list(ds["eval"])[:N_EVAL]


def encode(example):
    """Turn prompt + answer into tokens, and mark the prompt part 'do not grade'."""
    prompt = build_prompt(example["ticket"])
    full = prompt + example["label"] + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]

    # -100 means "do not grade this position": the ticket we gave the model.
    # TODO 3: build the labels: -100 for every ticket position, then the
    #         answer's own token numbers.
    labels = full_ids

    return {"input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels}


tokenized = ds.map(encode, remove_columns=ds["train"].column_names)

row = tokenized["train"][0]
n_ticket = sum(1 for l in row["labels"] if l == -100)
print(f"  First example: {n_ticket} tokens of ticket (not graded), "
      f"{len(row['labels']) - n_ticket} of answer (graded)")


# ==========================================================================
# STEP 2 - Train the adapter
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 2: Train the adapter (about 3 minutes)")
print("=" * 70)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    max_steps=MAX_STEPS,        # a fixed number of steps keeps the lab on time
    per_device_train_batch_size=8,
    learning_rate=1e-3,         # bigger than Lab 2's: only the small adapter changes
    warmup_steps=0.1,           # start with small adjustments for the first 10% of steps
    lr_scheduler_type="cosine", # ...then make them smaller again toward the end
    logging_steps=6,            # report the loss 4 times
    save_strategy="no",         # don't save extra copies while training
    use_cpu=True,               # run on the processor, not a graphics card
    optim="adamw_torch",
)

trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=tokenized["train"],
    # Pads each batch to the same length; the filler is not graded either.
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    processing_class=tokenizer,
)

# One plain line per loss report, plus the model's answer to one test ticket
# at that moment, so you can watch it learn.
plain_progress(trainer, watch=(tokenizer, eval_rows[0]))
t0 = time.time()
trainer.train()
elapsed = time.time() - t0


# ==========================================================================
# Save just the adapter
# ==========================================================================
peft_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
adapter_mb = sum(
    os.path.getsize(os.path.join(OUTPUT_DIR, f))
    for f in os.listdir(OUTPUT_DIR)
    if os.path.isfile(os.path.join(OUTPUT_DIR, f))
) / 1024 ** 2
# The model itself, without the adapters, at 4 bytes per number.
full_mb = sum(p.numel() * 4 for n, p in peft_model.get_base_model().named_parameters()
              if "lora_" not in n) / 1024 ** 2
print(f"\n  Trained in {elapsed:.0f} seconds. Saved adapter: {adapter_mb:.0f} MB "
      f"(a full copy: {full_mb:,.0f} MB).")


# ==========================================================================
# Results
# ==========================================================================
# Fill in a file while the trained model answers: each test ticket, with the
# best prompt's answer (Lab 1), Lab 2's and this lab's side by side.
saved = load_results()
ANSWERS_FILE = os.path.join(OUT_DIR, "lab3_answers.txt")
earlier = [("Lab 1 wrote", saved.get("bestprompt_smollm2", {}).get("answers", [])),
           ("Lab 2 wrote", saved.get("fullft_distilgpt2", {}).get("answers", []))]


def log(answers):
    write_answer_log(ANSWERS_FILE, eval_rows, earlier + [("Lab 3 wrote", answers)],
                     title="Each test ticket, its correct answer, and what each model "
                           "wrote back.\n"
                           "Lab 1 = SmolLM2 with the best prompt, no training.\n"
                           "Lab 2 = distilgpt2, fully trained.   "
                           "Lab 3 = SmolLM2 with a LoRA adapter.")


log([])
print("\n  Answering the test tickets, and writing out/lab3_answers.txt as it")
print("  goes (see labs.md). Which tickets did Lab 3 fix?")
preds = generate_predictions(peft_model, tokenizer, eval_rows, max_new_tokens=24,
                             on_answer=log)
s = score(preds, eval_rows)
save_result("lora_smollm2", s, {
    "model": BASE_MODEL, "trained": True, "method": "lora",
    "rank": peft_config.r, "alpha": peft_config.lora_alpha, "steps": MAX_STEPS,
    "train_seconds": round(elapsed), "artifact_mb": round(adapter_mb, 1),
    "answers": preds,
})

print("\n" + "=" * 70)
print(f"RESULTS ({N_EVAL} test tickets)")
print("=" * 70)
print_card([("best prompt", saved.get("bestprompt_smollm2", {})),
            ("Lab 2", saved.get("fullft_distilgpt2", {})), ("Lab 3", s)],
           note="Lab 3 trained about 2% of a model 4x bigger than Lab 2's.")
