"""
Lab 5 - Comparing everything you built.

Four models, one table. Plus a question people often forget to ask until it
causes problems later: did training on our one narrow task make the model
worse at everything else?

Run the two lm_eval commands from labs.md BEFORE running this script -- this
script reads their output.

STEP 6 is optional and needs a free GROQ_API_KEY. It compares everything you
built against a much larger hosted model (120 billion parameters), given the
best prompt we can write. It is skipped automatically if you have no key.
"""

import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common.quiet   # hide library warnings and progress bars

import torch
from collections import Counter, defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from common.triage import (
    load_tickets, generate_predictions, groq_predictions,
    score, print_scores, save_result, extract_fields, OUT_DIR, FRONTIER_MODEL,
    RANDOM_GUESS,
)

LORA_DIR = os.path.join(OUT_DIR, "smollm2-lora")
BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
N_EVAL = 20


# ==========================================================================
# STEP 1 - The comparison table: every run's scores side by side
# ==========================================================================
print("\n" + "=" * 78)
print("STEP 1: Every run's scores, side by side")
print("=" * 78)

with open(os.path.join(OUT_DIR, "results.json")) as f:
    results = json.load(f)

ORDER = ["baseline_distilgpt2", "bestprompt_smollm2", "fullft_distilgpt2",
         "lora_smollm2", "qlora_smollm2"]
LABEL = {
    "baseline_distilgpt2": "Lab 1  bare prompt distilgpt2  82M",
    "bestprompt_smollm2":  "Lab 1  best prompt SmolLM2  360M",
    "fullft_distilgpt2":   "Lab 2  retrain all distilgpt2  82M",
    "lora_smollm2":        "Lab 3  LoRA        SmolLM2  360M",
    "qlora_smollm2":       "Lab 4  QLoRA test  SmolLM2  360M",
}

hdr = (f"{'run':38s} {'format':>8s} {'category':>9s} {'priority':>9s} "
       f"{'queue':>7s} {'MB':>7s} {'steps':>6s} {'sec':>6s}")
print("\n" + hdr)
print("-" * len(hdr))
for key in ORDER:
    if key not in results:
        print(f"{LABEL[key]:38s}  (not run)")
        continue
    # TODO 1: print one table row per run -- format, category, priority,
    #         queue, file size (MB), step count, training seconds.
    r = results[key]
    print(f"{LABEL[key]:38s}")

print(f"  (random guessing, on average: category {RANDOM_GUESS['category_acc']:.1%}, "
      f"priority {RANDOM_GUESS['priority_acc']:.1%}, queue {RANDOM_GUESS['queue_acc']:.1%})")

print("""
  -> The row to beat is "best prompt". Read across every column: format,
     category, MB, steps and sec each answer a different question.
     Lab 4's row trained for only 4 steps, so its low scores show too little
     training -- not what 4 bits costs in quality.
""")


# ==========================================================================
# STEP 2 - Reload the best model for a closer look
# ==========================================================================
print("\n" + "=" * 78)
print("STEP 2: Reload the Lab 3 model to look at its mistakes")
print("=" * 78)

if not os.path.isdir(LORA_DIR):
    print(f"\n  No adapter found at {LORA_DIR}. STEPS 2-4 need the adapter Lab 3 saves:")
    print("  run Lab 3, then rerun this script.")
    raise SystemExit(0)

# Load the tokenizer we SAVED next to the adapter, not a fresh one -- it
# includes the padding-token change we made in Lab 3.
tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)

base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
# TODO 2: attach the Lab 3 adapter from LORA_DIR to the base model.
model = base
print(f"\n  Loaded {BASE_MODEL} + adapter from {LORA_DIR}")

eval_rows = list(load_tickets()["eval"])[:N_EVAL]
preds = generate_predictions(model, tokenizer, eval_rows, max_new_tokens=24)


# ==========================================================================
# STEP 3 - Score each category separately: where is it failing?
# ==========================================================================
print("\n" + "=" * 78)
print("STEP 3: Score each category separately")
print("=" * 78)

per = defaultdict(lambda: [0, 0])
confusions = Counter()
for pred, row in zip(preds, eval_rows):
    got = extract_fields(pred)["category"] or "<none given>"
    # TODO 3: count one attempt for this ticket's correct category, a hit
    #         when the model's answer matches, and otherwise the mix-up.
    per[row["category"]][1] += 1

print(f"\n  {'right category':24s} {'correct':>9s} {'tickets':>8s}")
print("  " + "-" * 44)
for cat in sorted(per):
    ok, n = per[cat]
    print(f"  {cat:24s} {ok:9d} {n:8d}")

if confusions:
    print("\n  Most common mix-ups (right category -> model's answer):")
    for (t, g), c in confusions.most_common(5):
        print(f"    {t:24s} -> {g:24s} x{c}")

print("\n  -> One overall score says IF there is a problem; this says WHERE. The usual")
print("     fix is more training examples of the weak category.")


# ==========================================================================
# STEP 4 - Look at actual failures
# ==========================================================================
print("\n" + "=" * 78)
print("STEP 4: Read the failures")
print("=" * 78)

shown = 0
for pred, row in zip(preds, eval_rows):
    if pred != row["label"] and shown < 4:
        more = "..." if len(row["ticket"]) > 100 else ""
        print(f"\n  TICKET   : {row['ticket'][:100]}{more}")
        print(f"  EXPECTED : {row['label']}")
        print(f"  PREDICTED: {pred!r}")
        shown += 1
if shown == 0:
    print("\n  No failures in this sample -- try raising N_EVAL.")

print("\n  -> Always read a few: they show whether the model misunderstands the task")
print("     or the correct answer itself is unclear.")


# ==========================================================================
# STEP 5 - Did training make the model worse at general tasks?
# ==========================================================================
print("\n" + "=" * 78)
print("STEP 5: Did training make the model worse at anything else?")
print("=" * 78)


def read_harness(pattern):
    files = sorted(glob.glob(os.path.join(OUT_DIR, pattern, "**", "results*.json"),
                             recursive=True))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)["results"]


base_res = read_harness("harness-base")
lora_res = read_harness("harness-lora")

if base_res and lora_res:
    tasks = sorted(set(base_res) & set(lora_res))
    print(f"\n  {'quiz':16s} {'before':>10s} {'+adapter':>10s} {'change':>10s}")
    print("  " + "-" * 50)
    for t in tasks:
        def pick(d):
            for k in ("acc_norm,none", "acc,none"):
                if k in d[t]:
                    return d[t][k]
            return None
        b, l = pick(base_res), pick(lora_res)
        if b is None or l is None:
            continue
        print(f"  {t:16s} {b:10.3f} {l:10.3f} {l - b:+10.3f}")
    print("""
  -> Two school-science quizzes, nothing to do with tickets. Expect changes
     near zero: with 20 questions, one answer moves a score by 0.05, so a
     small change either way is luck. A BIG drop means training hurt the
     model's other skills.
""")
else:
    print("\n  No quiz results found. Run the two lm_eval commands from labs.md first,")
    print("  then rerun this script.")


# ==========================================================================
# STEP 6 (OPTIONAL) - How does a large hosted model do with just a prompt?
# ==========================================================================
print("\n" + "=" * 78)
print("STEP 6 (optional, needs GROQ_API_KEY): A huge hosted model, prompt only")
print("=" * 78)

# The model name is set in common/triage.py. Groq retires hosted models from
# time to time; if this step says the model is not found, change it there.
FRONTIER = FRONTIER_MODEL               # about 330x the size of our SmolLM2

# TODO 4 (optional step): ask the large hosted model to answer each test
#         ticket. Show the first 2. Returns None if there is no GROQ_API_KEY.
groq_preds = None

if groq_preds is None:
    print("\n  Skipped -- no GROQ_API_KEY set. This step is optional. To run it, set a")
    print("  free key with scripts/setup-key.sh (see README.md) and rerun.")
else:
    gs = score(groq_preds, eval_rows)
    print_scores(f"PROMPT ONLY, NO TRAINING - {FRONTIER}", gs)
    save_result("zeroshot_groq_frontier", gs, {
        "model": FRONTIER, "trained": False, "method": "zero-shot prompt",
    })

    # Side by side with the model we fine-tuned.
    ours = results.get("lora_smollm2")
    if ours:
        print(f"\n  {'score':16s} {'LoRA 360M':>12s} {FRONTIER[:18]:>20s} {'difference':>10s}")
        print("  " + "-" * 62)
        for k, name in (("schema_valid", "right format"), ("category_acc", "category correct"),
                        ("priority_acc", "priority correct"), ("queue_acc", "queue correct")):
            print(f"  {name:16s} {ours[k]:11.1%} {gs[k]:19.1%} {gs[k] - ours[k]:+10.1%}")

    print("""
  -> A model about 330x bigger, with the best prompt we could write. If it wins,
     a prompt may be enough: run this check BEFORE investing in fine-tuning.
     (Even so: you pay per request, and every ticket leaves your network.)
""")


# ==========================================================================
# STEP 7 - What these results help you decide
# ==========================================================================
print("\n" + "=" * 78)
print("STEP 7: What you would tell your team")
print("=" * 78)
print("""
  Five questions you can now answer with evidence:
    Does fine-tuning help?         -> best-prompt row vs. trained rows
    Retrain the whole model?       -> Lab 2 vs. Lab 3 (scores, MB)
    Need a bigger model?           -> category scores by model size
    Did we break anything?         -> STEP 5
    Just use a hosted model?       -> STEP 6
""")
