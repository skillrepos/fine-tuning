"""
Lab 5 - Systematic evaluation.

Four models, one table. Plus the question nobody asks until it bites them in
production: did fine-tuning on our narrow task damage the model's general
ability?

Run the two lm_eval commands from labs.md BEFORE running this script -- this
script reads their output.

Step 6 is optional and needs a free GROQ_API_KEY. It compares everything you
built against a 120B hosted model given the best prompt we can write. It skips
cleanly if the key is absent.
"""

import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from collections import Counter, defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from common.triage import (
    load_tickets, generate_predictions, groq_predictions,
    score, print_scores, save_result, SCHEMA_RE, OUT_DIR, FRONTIER_MODEL,
)

LORA_DIR = os.path.join(OUT_DIR, "smollm2-lora")
BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
N_EVAL = 20


# ==========================================================================
# STEP 1 - The task-metric comparison table
# ==========================================================================
print("\n" + "=" * 78)
print("STEP 1: Task metrics across every run")
print("=" * 78)

with open(os.path.join(OUT_DIR, "results.json")) as f:
    results = json.load(f)

ORDER = ["baseline_distilgpt2", "fullft_distilgpt2", "lora_smollm2", "qlora_smollm2"]
LABEL = {
    "baseline_distilgpt2": "Lab 1  baseline    distilgpt2  82M",
    "fullft_distilgpt2":   "Lab 2  full FT     distilgpt2  82M",
    "lora_smollm2":        "Lab 3  LoRA fp32   SmolLM2  360M",
    "qlora_smollm2":       "Lab 4  QLoRA probe SmolLM2  360M",
}

hdr = (f"{'run':38s} {'schema':>8s} {'category':>9s} {'priority':>9s} "
       f"{'queue':>7s} {'MB':>7s} {'steps':>6s} {'sec':>6s}")
print("\n" + hdr)
print("-" * len(hdr))
for key in ORDER:
    if key not in results:
        print(f"{LABEL[key]:38s}  (not run)")
        continue
    # TODO 1: print one table row per run -- schema, category, priority,
    #         queue, artifact size, step count, training seconds.
    r = results[key]
    print(f"{LABEL[key]:38s}")

print("""
  Read the columns, not just the last row.

  schema   -- can the model produce the SHAPE we require at all?
  category -- does it understand the ticket?
  MB       -- what do we have to store and ship per fine-tuned variant?
  steps    -- how much supervision did it actually see?
  sec      -- what did it cost to produce?

  A single "accuracy" number would have hidden every one of those trade-offs.

  Now read rows 3 and 4 with the steps column in view. Lab 4 is a 4-step
  MECHANICS probe, not a quality run -- on CPU a like-for-like 24-step 4-bit
  run measures over an hour. So this table does NOT tell you what quantization
  costs in accuracy. It tells you something else worth just as much: the gap
  between rows 3 and 4 is almost entirely a STEP COUNT gap, which is the
  single most common reason a real fine-tune underperforms.

  The quantization-quality question is answered in the literature (QLoRA
  matches 16-bit LoRA on their benchmarks) and is set as homework at the end
  of Lab 4. What Lab 4 proved live is that 4-bit training runs at all, what it
  costs in memory, and how to export the result.
""")


# ==========================================================================
# STEP 2 - Reload the best model for a closer look
# ==========================================================================
print("\n" + "=" * 78)
print("STEP 2: Reload the LoRA model for error analysis")
print("=" * 78)

if not os.path.isdir(LORA_DIR):
    print(f"""
  No adapter found at {LORA_DIR}.

  Steps 2-4 need the artifact Lab 3 produces. Go back and run Lab 3, then rerun
  this script. Steps 1, 5 and 6 above and below do not need it.
""")
    raise SystemExit(0)

# Load the tokenizer we SAVED alongside the adapter, not a fresh one -- it
# carries the pad-token change we made in Lab 3.
tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)

base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
# TODO 2: attach the Lab 3 adapter from LORA_DIR to the base model.
model = base
print(f"\n  Loaded {BASE_MODEL} + adapter from {LORA_DIR}")

eval_rows = list(load_tickets()["eval"])[:N_EVAL]
preds = generate_predictions(model, tokenizer, eval_rows, max_new_tokens=24)


# ==========================================================================
# STEP 3 - Per-class accuracy: where exactly is it failing?
# ==========================================================================
print("\n" + "=" * 78)
print("STEP 3: Per-category breakdown")
print("=" * 78)

per = defaultdict(lambda: [0, 0])
confusions = Counter()
for pred, row in zip(preds, eval_rows):
    m = SCHEMA_RE.match(pred)
    got = m.group("category") if m else "<invalid>"
    # TODO 3: count one attempt for this true category, count a hit when
    #         the prediction matches, otherwise record the confusion pair.
    per[row["category"]][1] += 1

print(f"\n  {'true category':24s} {'correct':>9s} {'n':>4s}")
print("  " + "-" * 40)
for cat in sorted(per):
    ok, n = per[cat]
    print(f"  {cat:24s} {ok:9d} {n:4d}")

if confusions:
    print("\n  Most common confusions (true -> predicted):")
    for (t, g), c in confusions.most_common(5):
        print(f"    {t:24s} -> {g:24s} x{c}")

print("""
  An aggregate score tells you IF you have a problem. A per-class breakdown
  tells you WHERE. If one category is dragging everything down, the fix is
  usually more training examples for that category -- not more epochs.
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
        print(f"\n  TICKET   : {row['ticket'][:100]}")
        print(f"  EXPECTED : {row['label']}")
        print(f"  PREDICTED: {pred!r}")
        shown += 1
if shown == 0:
    print("\n  No failures in this sample -- try raising N_EVAL.")

print("""
  Always read a handful. Metrics are lossy summaries; the actual failures tell
  you whether the model is confused about the task or the data is ambiguous.
""")


# ==========================================================================
# STEP 5 - The general-capability regression check
# ==========================================================================
print("\n" + "=" * 78)
print("STEP 5: Did we damage the model? (lm-evaluation-harness results)")
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
    print(f"\n  {'task':16s} {'base':>10s} {'+adapter':>10s} {'delta':>10s}")
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
  This is the catastrophic-forgetting check, measured. The adapter made the
  model good at ticket triage; these numbers ask what that cost on everything
  else.

  How to read the deltas:

    near zero      -- what you should EXPECT here, and what we typically see.
                      24 steps of LoRA on 7 attention/MLP projections is a
                      very light touch. Nothing got overwritten.
    small positive -- noise, not improvement. At --limit 20 one extra correct
                      answer moves the number by 0.05. Do not go tell anyone
                      that fine-tuning on support tickets improved ARC.
    small negative -- also noise at this sample size, and normal even when
                      real.
    large negative -- the finding you ran this to catch. Learning rate or
                      epoch count too aggressive, or a dataset so narrow the
                      model collapsed onto one output shape. Mix in general
                      instruction data and retrain.

  The point of the check is not the number, it is having the number. Most
  teams ship a fine-tune having never measured this at all.

  Note: --limit 20 gives a deliberately noisy estimate so it finishes in a
  minute. It is enough to catch a collapse, not enough to publish. Drop
  --limit for a real number.
""")
else:
    print("""
  No harness output found. Run the two lm_eval commands from labs.md first:
  they write into out/harness-base/ and out/harness-lora/.
""")


# ==========================================================================
# STEP 6 (OPTIONAL) - What does a frontier model do with just a prompt?
# ==========================================================================
print("\n" + "=" * 78)
print("STEP 6: The prompt-engineering ceiling (optional -- needs GROQ_API_KEY)")
print("=" * 78)

# Imported from common/triage.py so there is ONE place to change when Groq
# retires a model -- which it does roughly twice a year. The model this workshop
# shipped with first, llama-3.3-70b-versatile, was decommissioned 2026-08-16.
FRONTIER = FRONTIER_MODEL               # ~330x the total parameters of our SmolLM2

# TODO 4 (optional step): ask the frontier model for a prediction on each
#         eval row. Show the first 2. Returns None with no GROQ_API_KEY.
groq_preds = None

if groq_preds is None:
    print("""
  Skipped -- no GROQ_API_KEY in the environment.

  This step is entirely optional and nothing else depends on it. If you have a
  free Groq account, add GROQ_API_KEY as a Codespaces secret, restart the
  Codespace, and rerun. Everything else in this workshop needs no accounts.
""")
else:
    gs = score(groq_preds, eval_rows)
    print_scores(f"ZERO-SHOT PROMPT - {FRONTIER}", gs)
    save_result("zeroshot_groq_frontier", gs, {
        "model": FRONTIER, "trained": False, "method": "zero-shot prompt",
    })

    # Head to head against the model we actually fine-tuned.
    ours = results.get("lora_smollm2")
    if ours:
        print(f"\n  {'metric':16s} {'LoRA 360M':>12s} {FRONTIER[:18]:>20s} {'delta':>10s}")
        print("  " + "-" * 62)
        for k in ("schema_valid", "category_acc", "priority_acc", "queue_acc"):
            print(f"  {k:16s} {ours[k]:11.1%} {gs[k]:19.1%} {gs[k] - ours[k]:+10.1%}")

    print("""
  Read this one carefully, because it is the question your management will ask.

  That model was given the BEST possible prompt -- every category, every
  priority, the full routing table, and the exact output contract. This is
  prompt engineering at its ceiling, on a model with roughly 330x the total
  parameters of the one you fine-tuned (120B, mixture-of-experts, about 5B of
  them active per token).

  Whichever way the accuracy falls, these do not change:

    Cost      -- you pay per token, on every request, forever. The prompt above
                 is ~250 tokens that get re-sent and re-billed every single time.
    Latency   -- a network round trip per ticket, versus local inference.
    Privacy   -- every ticket leaves your network. For a lot of enterprises that
                 ends the conversation on its own.
    Control   -- the model can be deprecated or silently updated underneath you.
                 Your adapter is a 37 MB file you own. This is not theoretical:
                 the model THIS STEP used to call, llama-3.3-70b-versatile, was
                 switched off by Groq on 2026-08-16, and the step broke until
                 someone edited one constant. Your Lab 3 adapter still runs.
    Drift     -- a prompt is a suggestion. Weights are a commitment.

  If the frontier model wins on accuracy here, the honest read is: this task is
  easy enough to prompt, and you should think hard about whether fine-tuning
  earns its keep. That is a legitimate outcome, and knowing it costs you one
  API call. Run this comparison BEFORE you invest in fine-tuning, not after.
""")


# ==========================================================================
# STEP 7 - The decision this all feeds
# ==========================================================================
print("\n" + "=" * 78)
print("STEP 7: What you would tell your team")
print("=" * 78)
print("""
  You now have the evidence to answer five separate questions:

    Does fine-tuning help this task?      -> baseline vs. any trained row
    Do we need full fine-tuning?          -> Lab 2 vs. Lab 3 quality per MB
    Do we need a bigger model?            -> category accuracy across sizes
    Did we break anything?                -> the harness deltas
    Should we have just used an API?      -> the Step 6 head-to-head

  That is the whole job. Everything else is hyperparameters.
""")
