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
  The row to beat is "best prompt": the best SmolLM2 did with no training.

  Read the columns, not just the last row.

  format   -- can the model write its answer in the required shape at all?
  category -- does it understand the ticket?
  MB       -- how much disk space each fine-tuned version takes
  steps    -- how many rounds of training it had
  sec      -- how long training took

  A single overall score would have hidden every one of those trade-offs.

  Now compare the last two rows, looking at the steps column. Lab 4 was a
  4-step test of how QLoRA works, not a quality run -- a full 24-step 4-bit
  run takes over an hour on a CPU. So this table does NOT show what 4-bit
  storage costs in quality. It shows something just as useful: the gap
  between the last two rows comes almost entirely from the number of
  training steps -- the most common reason a real fine-tune disappoints.

  The 4-bit quality question has been studied (the QLoRA paper found 4-bit
  matched 16-bit LoRA on its tests), and it is homework at the end of Lab 4.
  What Lab 4 showed live: 4-bit training works, how much memory it saves,
  and how to save the result as one standalone model.
""")


# ==========================================================================
# STEP 2 - Reload the best model for a closer look
# ==========================================================================
print("\n" + "=" * 78)
print("STEP 2: Reload the Lab 3 model to look at its mistakes")
print("=" * 78)

if not os.path.isdir(LORA_DIR):
    print(f"""
  No adapter found at {LORA_DIR}.

  STEPS 2-4 need the adapter Lab 3 saves. Go back and run Lab 3, then rerun
  this script. STEPS 1, 5 and 6 do not need it.
""")
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

print("""
  One overall score tells you IF there is a problem. Scoring each category
  separately tells you WHERE. If one category drags everything down, the
  usual fix is more training examples of that category -- not more passes
  over the same data.
""")


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

print("""
  Always read a few. Scores are summaries that hide detail; the actual
  mistakes show whether the model misunderstands the task or the correct
  answer itself is unclear.
""")


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
  These are two general-knowledge quizzes (school science questions) that
  have nothing to do with support tickets. Training on one narrow task can
  quietly make a model worse at everything else -- called "catastrophic
  forgetting". These numbers check for it.

  How to read the change column:

    near zero      -- what to expect here. 24 LoRA steps is a very light
                      touch; nothing got overwritten.
    small increase -- luck, not improvement. With only 20 questions, one
                      extra right answer moves the score by 0.05. Don't tell
                      anyone support-ticket training made it better at science.
    small decrease -- also luck at this size, and normal even when real.
    large decrease -- what this check is for. Training was too aggressive
                      (learning rate too high, too many passes) or the data
                      too narrow, and the model now gives one kind of answer
                      to everything. Mix in general examples and retrain.

  The value is in having the number at all. Many teams ship a fine-tuned
  model without ever checking this.

  Note: --limit 20 keeps this to about a minute, so the scores are rough. It
  is enough to catch a big drop, not enough to publish. Remove --limit for a
  real number.
""")
else:
    print("""
  No quiz results found. Run the two lm_eval commands from labs.md first:
  they save their results in out/harness-base/ and out/harness-lora/.
""")


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
    print("""
  Skipped -- no GROQ_API_KEY in the environment.

  This step is entirely optional and nothing else depends on it. If you have a
  free Groq account, set your key with scripts/setup-key.sh (see README.md)
  and rerun. Everything else in this workshop needs no accounts.
""")
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
  Read this one carefully -- it is the question your managers will ask.

  That model got the best prompt we could write: every category, every
  priority, the full routing table, and the exact answer format. It has
  about 330 times as many numbers as the model you fine-tuned (120 billion,
  though it uses only about 5 billion of them for each word it writes).

  However the scores compare, these points don't change:

    Cost      -- you pay for every word sent and received, on every request.
                 That prompt is about 250 tokens, sent and paid for every time.
    Speed     -- every ticket makes a trip over the internet, instead of
                 running on your own machine.
    Privacy   -- every ticket leaves your network. For many companies, that
                 alone rules it out.
    Control   -- the provider can retire or change the model at any time.
                 Your adapter is a 37 MB file you own. This really happened:
                 the model THIS STEP used to call, llama-3.3-70b-versatile,
                 was switched off by Groq on 2026-08-16, and the step broke
                 until someone changed one line. Your Lab 3 adapter still runs.
    Behavior  -- a prompt only asks the model to behave a certain way;
                 training changes the model itself.

  If the big model wins on the scores here, the honest conclusion is: this
  task is easy enough to solve with a prompt, so think hard about whether
  fine-tuning is worth it. That is a perfectly good outcome, and finding out
  costs one quick test. Run this comparison BEFORE you invest in
  fine-tuning, not after.
""")


# ==========================================================================
# STEP 7 - What these results help you decide
# ==========================================================================
print("\n" + "=" * 78)
print("STEP 7: What you would tell your team")
print("=" * 78)
print("""
  You now have the evidence to answer five separate questions:

    Does fine-tuning help this task?      -> best-prompt row vs. trained rows
    Do we need full fine-tuning?          -> Lab 2 vs. Lab 3: scores and file size
    Do we need a bigger model?            -> category scores by model size
    Did we break anything?                -> the quiz changes in STEP 5
    Should we just use a hosted model?    -> the STEP 6 comparison

  That is the whole job. Everything else is adjusting settings.
""")
