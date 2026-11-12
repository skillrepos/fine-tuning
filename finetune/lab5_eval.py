"""
Lab 5 - Compare everything, find where the model fails, and check that
training did not make it worse at everything else.

STEP 4 is optional and needs a free GROQ_API_KEY. It compares your model with
a much larger hosted model, given the best prompt we can write. Without a key
it is skipped.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TQDM_DISABLE", "1")   # no progress bars from the quiz tool
import common.quiet   # hide library warnings and progress bars

import contextlib, io, logging
import torch
from collections import Counter, defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from common.triage import (
    load_tickets, generate_predictions, groq_predictions, score, save_result,
    load_results, print_card, extract_fields, write_answer_log,
    OUT_DIR, FRONTIER_MODEL, RANDOM_GUESS,
)

LORA_DIR = os.path.join(OUT_DIR, "smollm2-lora")
BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
N_EVAL = 20
QUIZ = "arc_easy"        # a general school-science quiz, nothing to do with tickets
QUIZ_QUESTIONS = 20


# ==========================================================================
# STEP 1 - Every run side by side
# ==========================================================================
print("\n" + "=" * 78)
print("STEP 1: Every run side by side")
print("=" * 78)

results = load_results()
ROWS = [
    ("random guessing", RANDOM_GUESS),
    ("Lab 1 just the ticket, 82M", results.get("baseline_distilgpt2")),
    ("Lab 1 best prompt, 360M", results.get("bestprompt_smollm2")),
    ("Lab 2 retrain all, 82M", results.get("fullft_distilgpt2")),
    ("Lab 3 LoRA, 360M", results.get("lora_smollm2")),
]

print(f"\n  {'run':28s} {'category':>9s} {'priority':>9s} {'queue':>6s} "
      f"{'format':>7s} {'MB':>5s} {'sec':>5s}")
for name, r in ROWS:
    if r is None:
        print(f"  {name:28s}  (not run yet)")
        continue
    fmt = f"{r['schema_valid']:7.0%}" if "schema_valid" in r else f"{'-':>7s}"
    print(f"  {name:28s} {r['category_acc']:9.0%} {r['priority_acc']:9.0%} "
          f"{r['queue_acc']:6.0%} {fmt} {r.get('artifact_mb', '-'):>5} "
          f"{r.get('train_seconds', '-'):>5}")
print("\n  MB = size of what you would ship. sec = training time.")


# ==========================================================================
# STEP 2 - Where does the Lab 3 model go wrong?
# ==========================================================================
print("\n" + "=" * 78)
print("STEP 2: Where does the Lab 3 model go wrong?")
print("=" * 78)

if not os.path.isdir(LORA_DIR):
    print(f"\n  No adapter at {LORA_DIR}. Run Lab 3 first, then rerun this script.")
    raise SystemExit(0)

# Load the tokenizer saved next to the adapter: it has Lab 3's filler token.
tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
# TODO 1: attach the Lab 3 adapter (in LORA_DIR) to the base model.
model = base

eval_rows = list(load_tickets()["eval"])[:N_EVAL]

# Fill in a file while the model answers: every ticket it gets wrong.
MISTAKES_FILE = os.path.join(OUT_DIR, "lab5_mistakes.txt")


def log(answers):
    write_answer_log(MISTAKES_FILE, eval_rows, [("Lab 3 wrote", answers)],
                     title="Every test ticket the Lab 3 model got at least partly wrong.",
                     only_mistakes=True)


log([])
print("\n  Writing out/lab5_mistakes.txt as it answers: every ticket the model")
print("  gets wrong. Look at it while this step and the quiz run (see labs.md).")
preds = generate_predictions(model, tokenizer, eval_rows, max_new_tokens=24,
                             on_answer=log)

per = defaultdict(lambda: [0, 0])     # category -> [right, tickets]
mixups = Counter()                    # (right category, model's answer) -> count
for pred, row in zip(preds, eval_rows):
    got = extract_fields(pred)["category"] or "<none given>"
    # TODO 2: count one ticket for its right category; count it as right when
    #         the model's answer matches, otherwise count the mix-up.

print(f"\n  {'category':24s} right answers")
for cat in sorted(per):
    print(f"  {cat:24s} {per[cat][0]} of {per[cat][1]}")
if mixups:
    (right, got), n = mixups.most_common(1)[0]
    print(f"\n  Most common mix-up: {right} answered as {got} ({n}x)")


# ==========================================================================
# STEP 3 - Did training make the model worse at anything else?
# ==========================================================================
print("\n" + "=" * 78)
print(f"STEP 3: A general quiz ({QUIZ_QUESTIONS} questions), before and after training")
print("=" * 78)

# EleutherAI's lm-evaluation-harness: the standard tool for quizzing models.
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
logging.getLogger("lm-eval").setLevel(logging.ERROR)
logging.getLogger("lm_eval").setLevel(logging.ERROR)


def quiz(m):
    """Fraction of the quiz questions the model gets right."""
    with contextlib.redirect_stderr(io.StringIO()):   # hide the tool's progress bars
        lm = HFLM(pretrained=m, tokenizer=tokenizer, batch_size=1, device="cpu")
        out = simple_evaluate(model=lm, tasks=[QUIZ], limit=QUIZ_QUESTIONS,
                              verbosity="ERROR", log_samples=False)
    return out["results"][QUIZ]["acc,none"]


# Quiz the same model twice: first with the Lab 3 adapter switched off (the
# original model), then with it on again (the trained model).
# TODO 3: switch the adapter off for this one quiz, so it scores the original
#         model: put the line below inside  with model.disable_adapter():
before = quiz(model.get_base_model())
after = quiz(model.get_base_model())

print(f"\n  questions right: {before:.0%} before training, {after:.0%} after")
print("  -> One question is 5 points, so small changes are luck. A big drop would")
print("     mean training hurt the model's other skills.")


# ==========================================================================
# STEP 4 (optional, needs GROQ_API_KEY) - A huge hosted model, prompt only
# ==========================================================================
print("\n" + "=" * 78)
print("STEP 4 (optional): A huge hosted model, best prompt, no training")
print("=" * 78)

groq_preds = groq_predictions(eval_rows, model=FRONTIER_MODEL)
if groq_preds is None:
    print("\n  Skipped -- no GROQ_API_KEY set (see labs.md to add one).")
else:
    gs = score(groq_preds, eval_rows)
    save_result("zeroshot_groq_frontier", gs, {
        "model": FRONTIER_MODEL, "trained": False, "method": "zero-shot prompt",
    })
    print_card([("Lab 3 LoRA", results.get("lora_smollm2", {})), ("hosted", gs)],
               note=f"hosted = {FRONTIER_MODEL}, about 330x bigger than Lab 3's model.")
