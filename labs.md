# Fine-Tuning Small AI Models: LoRA, QLoRA & Evaluation

## Revision 1.7 - 11/02/26

**Workshop labs — 5 labs, 10-12 minutes each**

Everything in these labs runs on a **CPU-only, 4-vCPU GitHub Codespace**. No GPU,
no paid accounts, no API keys. Every model and dataset is public and free.

*(One exception: Lab 5 has a clearly-marked **optional** final step that compares
your fine-tuned model against a 120B hosted model on Groq. It needs a free Groq
API key — see "Optional: a free Groq API key" in `README.md` — and skips cleanly
without one. Nothing else in the workshop touches it.)*

**Not using the Codespace?** If you set up your own Mac or Windows machine with
`Local_Setup_Mac.md` or `Local_Setup_Windows.md`, keep that guide's **"What is
different from the Codespace"** section open alongside this document.

**Before Lab 1**, make sure your Codespace has finished building. You should see
a line like `>>> Model prefetch complete.` near the end of the terminal output.
If your terminal prompt does not start with `(.venv)`, run:

```
source .venv/bin/activate
```

**A note on the diff-merge steps.** Several steps use
`code -d <completed file> <your file>` to open VS Code's side-by-side diff view.
The **left** pane is the completed version; the **right** pane is your working
skeleton file, and it is the editable one. Click the small arrow (`→`) in the
gutter of each highlighted block to copy that block from left to right, then
save with `Ctrl+S` (`Cmd+S` on Mac).

Merge **every** highlighted block before running the file. VS Code often splits
one logical change into two adjacent regions — a deleted `# TODO` comment and
the inserted code — so work top to bottom until no highlighting remains rather
than counting blocks.

**Explanations while you merge.** Most blocks you merge come with a short
explanation of what the code does. When you see a small yellow comment icon in
the left gutter of the diff view, hover over the code beside it to read the note.
When **all** merges are done, save, then close the diff by clicking the `X` on
its tab.

When you first open a lab's starter file (before the diff), the places where
code still needs to go are highlighted in yellow — hover over one to see what
belongs there. Each highlight disappears once you have merged that block.

![Yellow highlight in a starter file](./images/ft-skeleton-highlight.png?raw=true "Yellow highlights mark where code still needs to go")

![Hover notes in the diff view](./images/merge-info3.png?raw=true "Hover over merge blocks to see what the code does")

**New to training terminology?** There is a one-page glossary at the end of this
document — loss, gradients, learning rate, batch, step, epoch, and friends. Keep
it open in a second tab if the vocabulary is new to you.

<br><br>

---

## **Lab 1 - Preparing a Dataset and Measuring a Baseline**

**Purpose: Turn raw support tickets into training data in the format the
training tools expect, see how a chat model wraps the same text, and measure
how badly an untrained model does at the task. Every later lab is measured
against the score you get here.**

**Estimated time: 10-12 minutes**

<br><br>

**Our task for the whole workshop.** Northwind Logistics wants every incoming
support ticket turned into one strict line:

```
category: access-management | priority: P2 | queue: identity-team
```

Eight categories, four priorities, six queues. This labeling scheme is
Northwind's own — no model has ever seen it — which is exactly when fine-tuning
beats writing a clever prompt.

<br><br>

1. Open a terminal in your Codespace (`Ctrl+Shift+` `` ` ``) and make sure you
are at the repository root. Then look at the raw data:

```
head -3 data/tickets_train.jsonl
wc -l data/tickets_train.jsonl data/tickets_eval.jsonl
```

You should see 281 training rows and 71 test rows, each with a `ticket` and its
correct `label`. The 71 test tickets are never used for training, so their
scores show how the model does on tickets it has not seen. (Real projects also
keep a third set for tuning decisions — more on that in Lab 5.)

<br><br>

2. Open the Lab 1 skeleton file so you can see what you are about to build:

```
code finetune/lab1_prep.py
```

Scroll through it. The four `# TODO` markers are where code is missing; the file
won't do anything useful until you merge in the finished code in the next step.

<br><br>

3. Open the diff view against the completed version and merge **every**
highlighted block from left to right, then save:

```
code -d extra/lab1_prep_complete.txt finetune/lab1_prep.py
```

![Diff-merge view for Lab 1](./images/ft-lab1-diff.png?raw=true "Merging the completed Lab 1 code")

<br><br>

4. Run it:

```
python finetune/lab1_prep.py
```

This takes about 25-40 seconds, mostly in STEP 7, where the model writes its
guesses. Warnings about `loss_type` or pad tokens are harmless.

<br><br>

5. Look at **STEP 1** and **STEP 2**. STEP 1 shows the data loaded as a training
set and a test set, with their columns and row counts. STEP 2 shows how many
tickets fall in each category. That balance matters: if one category had 80% of
the rows, the model would learn to guess that category for everything.

<br><br>

6. Look at **STEP 3**. This is what the model is trained on — the ticket, as the
*prompt*:

```
### Ticket:
Hi team, I cannot sign in to DriverPortal. It says my account is locked...

### Triage:
```

...and, kept separately, the answer it must produce (the *completion*). Keeping
them separate is what lets Lab 3 grade the model on the answer alone.

<br><br>

7. Look at **STEP 4**: the same ticket in a chat model's format, wrapped in
special markers like `<|im_start|>`. Each family of chat models uses its own
markers. Let the model's tokenizer (the tool that turns text into tokens) add
them for you with `apply_chat_template()` —
the wrong markers don't cause an error, just quietly worse answers.

<br><br>

8. Look at **STEP 5**. The tickets are short — around 50-70 tokens (word pieces)
each. Short examples are a big part of why this workshop can train on an
ordinary CPU; longer text gets slow very quickly.

<br><br>

9. Now the important part — **STEP 7 and STEP 8**. Read the three sample
predictions. distilgpt2 has never seen this format, so its answers are nowhere
close. Your scores should be:

```
  === BASELINE - distilgpt2, no training (n=20) ===
    schema valid :   0.0%
    category acc :   0.0%
    priority acc :   0.0%
    queue acc    :   0.0%
```

![Lab 1 baseline scores](./images/ft-lab1-baseline.png?raw=true "Baseline scores are zero across the board")

Why *exactly* zero? distilgpt2 is a **base** model: it was only trained to
continue text, never to follow instructions, so it just keeps writing from
wherever the prompt stops. The SmolLM2-**Instruct** model in Lab 3 was also
trained to follow instructions — keep that difference in mind when you compare
labs.

<br><br>

10. **Discussion — why not just write a better prompt?** You could list every
category, priority and queue in the prompt. A large hosted model might follow
it; an 82-million-parameter model can't follow a multi-part instruction
reliably. And with any model, that long prompt is sent — and paid for — on every
request. Fine-tuning builds the knowledge into the model instead: shorter
prompts, faster and cheaper calls, and a model small enough to run where you
need it.

Keep the baseline in mind: everything from here is measured against zero.

<br><br>

11. **Optional (about 2 minutes).** Change `N_EVAL = 20` near the top of
`finetune/lab1_prep.py` to `40` and rerun. Still zero, but it takes about twice
as long: more test tickets give a more reliable score, at the cost of time. Put
it back to `20` before you move on.

<p align="center">**[END OF LAB]**</p></br></br>

---

## **Lab 2 - Full Fine-Tuning with the Hugging Face Trainer**

**Purpose: Retrain every part of the small distilgpt2 model with Hugging Face's
standard `Trainer`, and measure what that costs in time and disk space. Labs 3
and 4 are compared against this.**

**Estimated time: 10-12 minutes**

<br><br>

1. Open the Lab 2 skeleton and read through it before merging:

```
code finetune/lab2_fullft.py
```

There are four `# TODO` markers. Again — this file won't train correctly until
you merge.

<br><br>

2. Merge in the completed code — work top to bottom until no highlighting is
left — and save:

```
code -d extra/lab2_fullft_complete.txt finetune/lab2_fullft.py
```

<br><br>

3. Run it. **Training takes about 2-4 minutes** — start it now and read the
notes below while it runs:

```
python finetune/lab2_fullft.py
```

<br><br>

4. Look at the **STEP 1** output. Every one of distilgpt2's 82 million
parameters — the numbers inside the model that training adjusts — will be
trained: `100.0000%`. Remember that number; in Lab 3 the same line reads about
**2.3%**.

<br><br>

5. Read the `<|pad|>` note in **STEP 1**. It avoids a common bug. Training
examples have different lengths, so shorter ones are filled out with a
*padding* token. GPT-2 models have none, and many tutorials reuse the *end*
token for it. But padding is left out of the grading — so if padding and end
share a token, the real end tokens are left out too, and the model never learns
to stop. Two different tokens avoid that.

<br><br>

6. Look at **STEP 3**. Compare the end of the input with the end of the labels
(what the model is graded on):

```
  input_ids tail : [..., 50257, 50257, 50257]
  labels tail    : [...,  -100,  -100,  -100]
```

`-100` means "don't grade this position": the padding is ignored, while the
real end token is kept.

<br><br>

7. Watch the training loss — how wrong the model's guesses are — fall from
about 5 at the start to below 1 by the end. If it stays flat or jumps around, the learning
rate (step size) is usually the problem. Note that a falling loss only shows the
model is learning its training examples; the test scores in STEP 7 show whether
that works on new tickets. (The "Final training loss" line is the average over
the whole run, so it is higher than the last few values.)

![Lab 2 training loss falling](./images/ft-lab2-training.png?raw=true "Training loss during full fine-tuning")

<br><br>

8. When training finishes, note two numbers from **STEP 5 and STEP 6**: how long
training took, and the size of the saved model (a few hundred MB). Retraining
everything produces a whole new copy of the model — ten customers means ten
copies.

<br><br>

9. Look at **STEP 7**. Expect `schema valid` (answers in the right format) to
jump dramatically while category accuracy lags behind. That gap is the lesson:
**learning a format is easy; learning to pick the right category takes a more
capable model.**

<br><br>

10. **Discussion — when would you retrain everything?** When you have a very
large dataset (tens of thousands of examples), need the model to learn genuinely
new abilities rather than a new output style, and can afford the time and the
storage for a full copy per version. For most business work, the next two labs
are the better trade.

<br><br>

11. **Optional (about 2 minutes of training).** Change `num_train_epochs=2` to
`1` (one pass over the training data instead of two), rerun, and compare STEP 7
with your first run. You changed one thing and kept everything else the same —
that is how to run a fair comparison. Then put it back to `2` and rerun, so
Lab 5 compares the standard run.

<p align="center">**[END OF LAB]**</p></br></br>

---

## **Lab 3 - LoRA Fine-Tuning with PEFT**

**Purpose: Fine-tune a model four times bigger than Lab 2's while training only
about 2% of it, using LoRA. Grade the model only on its answer. Save the result
as a small add-on file instead of a whole new copy of the model.**

**Estimated time: 10-12 minutes**

<br><br>

**The idea in plain words.** Lab 2 retrained every number inside the model.
LoRA leaves the model untouched and trains a small add-on beside it, called an
*adapter*, that adjusts its answers. The adapter is about 2% the size of the
model, so training needs far less memory and the saved result is tens of
megabytes, not a whole new model. (The math is on the LoRA slides.)

<br><br>

1. Open the Lab 3 skeleton:

```
code finetune/lab3_lora.py
```

<br><br>

2. Merge in the completed code — work top to bottom until no highlighting is
left — and save:

```
code -d extra/lab3_lora_complete.txt finetune/lab3_lora.py
```

<br><br>

3. Run it. **Budget about 5 minutes** — about 1 minute to load the model,
3 minutes of training, and 1 minute to test the result:

```
python finetune/lab3_lora.py
```

You may see a line mentioning `gemm_4bit_forward` and `kernels` — it is a
harmless message from the 4-bit library (it appears in Labs 3-5). Ignore it.

<br><br>

4. Look at **STEP 2**. It lists the layers inside this model where LoRA can
attach adapters. The names differ between kinds of model, so always use the
ones the model itself reports — a list copied from another model can quietly
miss layers.

<br><br>

5. Look at **STEP 3 and STEP 4**. Two settings worth knowing:

- **`r` (rank)** — the size of the adapter. Bigger can learn more, but trains
  slower. 8-16 suits a new format or style; 32-64 suits new behavior.
- **`lora_alpha`** — how strongly the adapter's adjustments count. The usual
  rule is twice `r` (here 16 and 32).

Then find the line STEP 4 prints: `trainable%: 2.34` or close to it. That is
how much of the model will be trained. Lab 2 trained 100%. (STEP 1 also shows
100%, because that is before the adapters are added.)

![Lab 3 trainable parameter percentage](./images/ft-lab3-trainable.png?raw=true "LoRA trains about 2% of the parameters")

<br><br>

6. Look at **STEP 5**. This is the other big change from Lab 2: the model is
now graded only on its answer. In the training data, every position that
belongs to the ticket is marked `-100`, which means "don't grade this":

```
labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
```

The script prints something like `53 tokens, 36 from the prompt (not graded),
17 graded`. In Lab 2 the model was also graded on repeating the ticket back —
wasted effort for a small model.

<br><br>

7. Watch the training loss — how wrong the model's guesses are — fall as it
trains. This lab uses a bigger learning rate (step size) than Lab 2, `1e-3`
instead of `3e-4`, which is safe because only the small adapter is changing.

<br><br>

8. Look at **STEP 7**. The saved adapter is a few tens of MB; a full copy of the
model would be about 1.4 GB. That is the practical payoff: one shared model,
plus one small adapter file per task or customer.

<br><br>

9. Look at **STEP 8** and compare the scores with Lab 2's — category accuracy
should be clearly better. (The model changed too, so this shows the workshop's
progress, not how much LoRA alone helped.)

<br><br>

10. **Optional — change one thing and rerun:**

- **Smaller adapter (about 4 minutes).** In STEP 3 of your file, change `r=16`
  to `r=8` and `lora_alpha=32` to `16`. The trained percentage should halve and
  the adapter file shrink; STEP 8 shows whether the scores suffer.
- **DoRA (about 10 minutes on a CPU — best done at home).** Set
  `use_dora=True`. DoRA is a LoRA variation that often does a little better
  with small adapters, but both training and testing are noticeably slower.

Lab 5 uses the adapter this lab saves, so put the settings back and rerun
before you move on.

<br><br>

11. **Discussion — is LoRA always the better choice?** For teaching a model a
new format, style or task, usually yes, and it is far cheaper. Retraining the
whole model can still win when the model must absorb a lot of new knowledge.

<p align="center">**[END OF LAB]**</p></br></br>

---

## **Lab 4 - QLoRA: 4-bit Quantization, Merging, and Export**

**Purpose: Store the model in a compressed 4-bit form to save memory, show that
an adapter can still be trained on top of it, measure what that costs on a CPU,
and combine the adapter and the model into one file you can ship.**

**Estimated time: 10-12 minutes**

<br><br>

**The idea in plain words.** LoRA trains only a small adapter, but the whole
model still has to sit in memory while you train. QLoRA shrinks that model by
storing each of its numbers in 4 bits instead of 16 or 32 — a bit like saving a
photo at lower quality — and then trains a LoRA adapter on top, as before. The
goal is to fit a bigger model into the memory you have.

**Read this before you run it.** This lab shows *how* QLoRA works and how much
memory it saves — not how good its results are. It trains for only **4 steps**,
because on a CPU the 4-bit math is very slow: about 20 times slower per example
than Lab 3. (On a GPU the slowdown is much smaller.) **QLoRA saves memory, not
time.**

<br><br>

1. Open the Lab 4 skeleton:

```
code finetune/lab4_qlora.py
```

<br><br>

2. Merge in the completed code — work top to bottom until no highlighting is
left — and save:

```
code -d extra/lab4_qlora_complete.txt finetune/lab4_qlora.py
```

<br><br>

3. Run it. **This one takes about 6-7 minutes** — the 4 training steps take
about 3 of them, and STEP 8 loads a second model:

```
python finetune/lab4_qlora.py
```

<br><br>

4. Look at **STEP 1 and STEP 2** — the same model stored three ways:

```
  bfloat16   :     ~690 MB   (2 bytes per weight)
  float32    :   ~1,380 MB   (4 bytes per weight -- what Lab 3 used)
  4-bit NF4  :     ~240 MB
```

4-bit isn't a full 8 times smaller than 32-bit: some parts of the model stay
uncompressed. The bigger the model, the closer you get to the full saving.

<br><br>

5. Look at **STEP 3**, which looks inside one compressed layer. Three things to
notice:

- **Packed** — two 4-bit numbers are stored in each byte.
- **In groups** — numbers are compressed 64 at a time, each group with its own
  scale, so one unusual value can't spoil the rest.
- **NF4** — the 4-bit format QLoRA uses. Its 16 possible values are spaced to
  match how a trained model's numbers are actually spread out, so it loses
  less accuracy than evenly spaced values would.

![Lab 4 memory comparison and Linear4bit inspection](./images/ft-lab4-memory.png?raw=true "4-bit NF4 memory footprint")

<br><br>

6. Look at **STEP 5**. The loss changes — proof that training still reaches the
adapter even though the model underneath is compressed. Then read the timing:

```
  4-bit NF4, batch 2       : ~45-50 s/step
  Lab 3 float32, batch 8   : ~6-10 s/step
```

Lab 3 also handled four times as many examples per step (8 instead of 2), so per
example, 4-bit is roughly 20-30 times slower here. **Takeaway:** use QLoRA when memory
is the problem. If the model already fits, plain LoRA trains faster.

<br><br>

7. Look at **STEP 6**, the merge: folding the adapter permanently into the model,
so you can ship one ordinary model. Read the code: we trained on the 4-bit
model, but merge into a fresh full-precision (bfloat16) copy. Merging into the
compressed version would make its accuracy loss permanent. The order is always:
train compressed → merge into full precision → compress the result again if
you need to.

<br><br>

8. Merge, or keep the adapter separate? A merged model is an ordinary model
that needs nothing extra to run, but you lose Lab 3's one-model, many-adapters
trick. Merge for a single dedicated use; keep adapters separate to serve many
customers or tasks.

<br><br>

9. Look at **STEP 7**, which scores the merged model — the file you would ship.
The scores are poor, as expected after only 4 training steps. That shows how
much the amount of training matters, not that QLoRA is worse than LoRA.

A fair test (same model, data and settings, 24 steps) takes about 70 minutes on
a CPU, so it is homework: set `QLORA_STEPS = 24` and `QLORA_BATCH = 8`, run it,
and compare with Lab 3's results in `out/results.json`. Published research
found almost no difference in quality.

<br><br>

10. Look at **STEP 8**. TinyLlama, a 1.1-billion-parameter model, loaded in 4-bit
takes less memory than Lab 3's 360-million-parameter model did at full size.
That is the point of QLoRA: on the same hardware, you can fine-tune a model
about one size class bigger.

<br><br>

11. **Optional, at home, on a GPU.** Point `BASE_MODEL` at TinyLlama and nothing
else needs to change. For `"microsoft/Phi-3-mini-4k-instruct"` (3.8 billion
parameters), also change `target_modules` to
`["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]` — Phi-3 uses different
layer names (see Lab 3, STEP 2). It is a 7.6 GB download, which is why it isn't
a live lab.

<p align="center">**[END OF LAB]**</p></br></br>

---

## **Lab 5 - Systematic Evaluation**

**Purpose: Put all four runs side by side, see which kinds of tickets the model
gets wrong, and check whether fine-tuning damaged the model's general
abilities.**

**Estimated time: 10-12 minutes**

<br><br>

1. First, test the original SmolLM2 model on a standard set of general-knowledge
questions, using EleutherAI's lm-evaluation-harness — the standard tool for
this. It saves its results to a file the Lab 5 script reads. About 1-2 minutes:

```
lm_eval run --model hf \
  --model_args pretrained=HuggingFaceTB/SmolLM2-360M-Instruct,dtype=float32 \
  --tasks arc_easy,arc_challenge \
  --device cpu \
  --limit 20 \
  --batch_size 1 \
  --output_path out/harness-base
```

Warnings about the number of "few-shot" examples are harmless.

<br><br>

2. Now run the same test **with your Lab 3 adapter attached** — the `peft=` part
adds it. This needs `out/smollm2-lora` from Lab 3; if Lab 3 didn't finish, go
back and complete it first. Another 1-2 minutes:

```
lm_eval run --model hf \
  --model_args pretrained=HuggingFaceTB/SmolLM2-360M-Instruct,peft=out/smollm2-lora,dtype=float32 \
  --tasks arc_easy,arc_challenge \
  --device cpu \
  --limit 20 \
  --batch_size 1 \
  --output_path out/harness-lora
```

<br><br>

3. Open the Lab 5 skeleton:

```
code finetune/lab5_eval.py
```

<br><br>

4. Merge in the completed code — work top to bottom until no highlighting is
left — and save. There are four `# TODO` markers; the last one is for the
optional Groq step and is fine to merge whether or not you have a key:

```
code -d extra/lab5_eval_complete.txt finetune/lab5_eval.py
```

<br><br>

5. Run it:

```
python finetune/lab5_eval.py
```

<br><br>

6. Look at **STEP 1**, the comparison table:

```
run                                      schema  category  priority   queue      MB  steps    sec
-------------------------------------------------------------------------------------------------
Lab 1  baseline    distilgpt2  82M        0.0%     0.0%     0.0%   0.0%       -      -      -
Lab 2  full FT     distilgpt2  82M        ...      ...      ...    ...      ...     72    ...
Lab 3  LoRA fp32   SmolLM2  360M          ...      ...      ...    ...      ...     24    ...
Lab 4  QLoRA probe SmolLM2  360M          ...      ...      ...    ...      ...      4    ...
```

Read every column, not just one. `schema`: does it answer in the right format?
`category`: does it understand the ticket? `MB`: how big is what you would ship?
`steps`: how much training did it get? `sec`: how long did training take? A
single "accuracy" number would hide all of that.

Row 4 (Lab 4) trained for only 4 steps, so its low scores show what too little
training does — not what 4-bit costs in quality.

![Lab 5 comparison table](./images/ft-lab5-table.png?raw=true "Comparison table across all four runs")

<br><br>

7. Look at **STEP 3**. It breaks the results down by category and lists the most
common mix-ups (true category → predicted category). An overall score tells you
*whether* there is a problem; this tells you *where*. When one category drags the
average down, the usual fix is more training examples for that category.

<br><br>

8. Look at **STEP 4** and read the actual failures. Four wrong answers tell you
in seconds whether the model is confused or your labels are unclear — two
different problems with different fixes.

<br><br>

9. Look at **STEP 5**, the general-knowledge scores before and after training:

```
  task                   base   +adapter      delta
  --------------------------------------------------
  arc_easy              0.400      0.450     +0.050
  arc_challenge         0.250      0.250     +0.000
```

Your adapter made the model good at ticket triage; this checks what that cost on
everything else. Expect changes near zero — 24 steps of LoRA is a light touch. A
small gain is just noise: with 20 questions, one answer moves the score by 0.05.

What you are watching for is a **big drop**, called *catastrophic forgetting*:
the model got worse at everything except your task. The usual fix is gentler
training, or mixing some general examples into the training data.

<br><br>

10. Two honesty notes. First, 20 questions per test can catch a big drop but
can't give a reliable score; remove `--limit 20` for a real number (much slower
on a CPU). Second, these labs reuse Lab 1's 71 test tickets throughout. In real
projects, keep a separate set for tuning decisions and use the test set once, at
the end — a test set you keep adjusting against stops being a fair test.

<br><br>

11. **Optional — compare with a big hosted model.** **STEP 6** sends the same 20
tickets to `openai/gpt-oss-120b`, a very large model hosted by Groq, with the
best prompt we could write (every category, priority and routing rule), and
compares the result with your LoRA model.

This needs a free Groq API key. If you haven't set one up, follow "Optional: a
free Groq API key" in `README.md` (about two minutes), then run
`python finetune/lab5_eval.py` again. **Without a key, the step prints a note and
skips** — nothing else depends on it.

> **If this step reports the model was not found ("404"),** Groq has retired it
> — that happens a few times a year. Run `python scripts/check_groq.py` to see
> which models your key can use, and put one of them in `FRONTIER_MODEL` in
> `common/triage.py`.

Whatever the result, with a hosted model you pay for every request, every
ticket leaves your company, and the provider can retire the model. And if the
big model wins, the task may not need fine-tuning at all — a useful finding.
**Run this comparison before you invest in fine-tuning, not after.**

<br><br>

12. **Discussion — the five questions you can now answer with evidence.**

| Question | Where the answer is |
|---|---|
| Does fine-tuning help this task at all? | Lab 1 baseline vs. any trained row |
| Do we need to retrain the *whole* model? | Lab 2 vs. Lab 3: results per MB and per second |
| Do we need a bigger model? | category accuracy across the model sizes |
| Did we break anything? | the STEP 5 before/after scores |
| Should we have just used a hosted model? | the STEP 6 comparison |

Close the loop the way you would at work: in one or two sentences, write the
recommendation you would send Northwind Logistics — which model (if any) goes to
production, and the two numbers from today you would cite to defend it.

<p align="center">**[END OF LAB]**</p></br></br>

---

## Where to go next

- **More and better data.** 281 examples is a demo. Real projects typically use
  1,000-10,000 carefully checked examples — and quality matters more than count.
- **Teaching preferences.** Once a model gets the format right, techniques such
  as DPO and GRPO (both in Hugging Face's TRL library) teach it which of several
  correct answers people prefer.
- **Serving many adapters.** Serving tools such as vLLM and TGI can keep one
  model in memory and swap in a different adapter for each request.
- **Running a merged model elsewhere.** A merged model can be converted for
  tools such as Ollama and `llama.cpp`, or compressed further for fast serving.
- **Using a GPU.** These scripts run unchanged on a GPU: remove `use_cpu=True`
  and they run 50-100x faster on a single consumer graphics card.

<br><br>

---

## Appendix — one-page glossary

Plain-language versions of every training term used today.

| Term | What it means here |
|---|---|
| **Parameter (weight)** | One of the millions of numbers inside a model. Training adjusts them; "82M parameters" means 82 million of them. |
| **Loss** | A single number scoring how wrong the model's predictions are on a batch. Training exists to push it down. |
| **Gradient** | For each parameter, which direction (and how strongly) to nudge it to reduce the loss. |
| **Optimizer** | The code that applies those nudges. Ours, called **Adam**, also keeps two extra numbers for every parameter it trains — one reason retraining a whole model needs so much memory. |
| **Learning rate** | How big each nudge is. Too high: the loss jumps around. Too low: nothing happens. |
| **Batch** | How many examples are processed together for one update. |
| **Step** | One update of the weights (one batch). |
| **Epoch** | One full pass over the training data. 281 examples at batch 8 ≈ 35 steps per epoch. |
| **Token** | The unit models read — roughly a word fragment. Our tickets are ~50-70 tokens. |
| **Tokenizer** | The tool that turns text into tokens (and back), and adds a chat model's special markers. |
| **EOS / PAD** | End-of-sequence token (the stop signal the model must learn) / filler that makes batch rows the same length. They must be *different* tokens — see Lab 2. |
| **Masking / -100** | Marking positions with -100 so they aren't graded — used so the model is graded only on its answer, not on repeating the ticket. |
| **Base vs. instruct model** | A base model only continues text. An instruct model had a further training pass to follow instructions. distilgpt2 is base; SmolLM2-360M-Instruct is instruct. |
| **Rank (r)** | The size of a LoRA adapter. Bigger can learn more, but is bigger and slower to train. |
| **Alpha (lora_alpha)** | How strongly the adapter's adjustments count. The usual setting is twice r. |
| **Adapter** | The small add-on LoRA trains (tens of MB), saved separately from the unchanged model. |
| **Quantization / NF4** | Storing a model's numbers in fewer bits to save memory. NF4 is the 4-bit format QLoRA uses. |
| **Merge** | Folding an adapter back into the base weights to get one ordinary model. Always merge into a full-precision base. |
| **Held-out / eval set** | Data the model never trained on. The only scores that count. |
| **Baseline** | The untrained model's score — what every later run is measured against. Ours is 0%. |
| **Catastrophic forgetting** | Training hard on a narrow task degrading everything else the model could do. Lab 5's harness run is the check. |

<br><br>

---

<p align="center">
<b>For educational use only by the attendees of our workshops.</b>
</p>
<p align="center">
<b>(c) 2026 Tech Skills Transformations and Brent C. Laster. All rights reserved.</b>
</p>
