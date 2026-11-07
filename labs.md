# Fine-Tuning Small AI Models: LoRA, QLoRA & Evaluation

## Revision 1.12 - 11/07/26

**Workshop labs**


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

**Purpose: Turn raw support tickets into training data, and measure the best
an untrained model can do. Every later lab is measured against that score.**

**Estimated time: 10-12 minutes**

<br><br>

**Our task for the whole workshop.** Northwind Logistics wants every incoming
support ticket turned into one strict line:

```
category: access-management | priority: P2 | queue: identity-team
```

Eight categories, four priorities, six queues — a labeling scheme no model has
ever seen.

<br><br>

1. Go to a terminal in your Codespace and make sure you
are at the repository root (/workspaces/fine-tuning). Then look at the raw data:

```
head -3 data/tickets_train.jsonl
wc -l data/tickets_train.jsonl data/tickets_eval.jsonl
```

281 training rows and 71 test rows, each a `ticket` and its correct `label`.
The test tickets are never used for training.

![Training dataset](./images/ft-1-1.png?raw=true "Training dataset")

<br><br>

2. Open the Lab 1 skeleton file so you can see what you are about to build:

```
code finetune/lab1_prep.py
```

The four `# TODO` markers are where code is missing.

<br><br>

3. Open the diff view against the completed version and merge **every**
highlighted block from left to right, then close the diff tab to save the changes:

```
code -d extra/lab1_prep_complete.txt finetune/lab1_prep.py
```

![Diff-merge view for Lab 1](./images/ft-lab1-diff.png?raw=true "Merging the completed Lab 1 code")

<br><br>

4. Run it (about 2 minutes). The output is in *STEP* sections, which the next
lab steps walk through:

```
python finetune/lab1_prep.py
```

<br><br>

5. Look at **STEP 1** and **STEP 2**: the data loaded, then a count of each
category and priority. If one category had 80% of the rows, the model would
learn to guess it for everything.

![First steps output](./images/ft-1-5.png?raw=true "First steps output")

<br><br>

6. Look at **STEP 3**. The ticket, inside a fixed template, is the *prompt*;
the row's label is the *completion* the model must produce. Keeping the two
apart lets Lab 3 grade only the answer.

![Step 3 output](./images/ft-1-6.png?raw=true "Step 3 output")

<br><br>

7. Look at **STEP 4**: the same ticket in a chat model's format, wrapped in
markers like `<|im_start|>`. Each model family has its own markers;
`apply_chat_template()` adds the right ones.

![Output with markers](./images/ft-1-7.png?raw=true "Output with markers")

<br><br>

8. Look at **STEP 5**. The tickets are short — around 50-70 tokens (word
pieces) each — which is why this workshop can train on an ordinary CPU.

<br><br>

9. Now **STEP 7 and STEP 8**. STEP 7 gives distilgpt2 20 test tickets with
just the bare ticket prompt — compare each `EXPECTED` line with `PREDICTED`.
STEP 8 scores all 20, next to what random guessing would score:

```
  === BASELINE - distilgpt2, no training (20 tickets) ===
    right format     :   0.0%
    category correct :   0.0%
    priority correct :   0.0%
    queue correct    :   0.0%

  === RANDOM GUESSING (average, no model at all) ===
    right format     :      -
    category correct :  12.5%
    priority correct :  25.0%
    queue correct    :  16.7%
```

![Lab 1 baseline scores](./images/ft-lab1-baseline.png?raw=true "Bare-prompt scores next to random guessing")

`right format` requires the first line to be exactly
`category: ... | priority: ... | queue: ...`; the other three count the right
answer anywhere, in any form. distilgpt2 is a **base** model, trained only to
continue text, so it never names a category. 0% means nobody told it the task —
a poor bar.

<br><br>

10. Look at **STEP 9**, the fair bar: SmolLM2-360M-**Instruct** (the model
Labs 3 and 4 train) given the best prompt we could write — every label, the
routing rules, the format, eight worked examples — and no training:

```
  score                random  bare prompt  best prompt
  ------------------------------------------------------
  right format              -           0%         100%
  category correct        12%           0%          30%
  priority correct        25%           0%          45%
  queue correct           17%           0%          45%
```

![Lab 1 best-prompt baseline](./images/ft-lab1-bestprompt.png?raw=true "Random guessing, bare prompt and best prompt side by side")

The format is now perfect, but fewer than 1 category in 3 is right. **The
best-prompt column is the score every later lab has to beat.**

<br><br>

11. **Discussion — then why fine-tune at all?** That prompt is about 700
tokens, sent and paid for on every request, against about 50 for the bare
prompt. Fine-tuning builds that knowledge into the model instead — if it can
beat the best prompt. That is the question for the rest of the workshop.

<br><br>

12. **Optional (about 4 minutes).** Change `N_EVAL = 20` near the top of
`finetune/lab1_prep.py` to `40` and rerun: more test tickets, a more reliable
score, more time. Then put it back to `20` and rerun, so Lab 5 compares
20-ticket scores.

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

5. STEP 1 also prints two token numbers: the *padding* token (filler that
makes a group of examples the same length) and the *end-of-text* token (the
stop signal). GPT-2 has no padding token, and many tutorials reuse the
end-of-text token for it. But padding is left out of the grading — so the model
would never be graded on stopping, and never learn to. Two different tokens
avoid that.

<br><br>

6. Look at **STEP 3**. It shows where the shortest example ends and its
padding begins, and which tokens are graded:

```
    tokens       : [..., 50256, 50257, 50257, ...]
    what's graded: [..., 50256,  -100,  -100, ...]
```

`-100` means "skip this one when grading": the padding (50257) is skipped,
while the end-of-text token (50256) is still graded.

<br><br>

7. Watch the training loss — how wrong the model's guesses are — fall from
about 5 at the start to below 1 by the end. If it stays flat or jumps around,
the learning rate (step size) is usually the problem. A falling loss only shows
the model is learning its training examples; the test scores in STEP 7 show
whether that works on new tickets.

![Lab 2 training loss falling](./images/ft-lab2-training.png?raw=true "Training loss during full fine-tuning")

<br><br>

8. When training finishes, note two numbers from **STEP 5 and STEP 6**: how long
training took, and the size of the saved model (a few hundred MB). Retraining
everything produces a whole new copy of the model — ten customers means ten
copies.

<br><br>

9. Compare **STEP 7** with Lab 1's best-prompt row, the score to beat. The
trained 82M model wins on category and queue — **a small trained model beats a
bigger prompted one.** Priority doesn't move.

![Lab 2 scores](./images/ft-lab2-scores.png?raw=true "Lab 2 scores after full fine-tuning")

<br><br>

10. **Discussion — when would you retrain everything?** When you have a very
large dataset, need the model to learn new abilities rather than a new output
style, and can afford a full copy per version. For most business work, the
next two labs are the better trade.

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
LoRA leaves the model untouched and trains a small add-on beside it, an
*adapter*, about 2% of the model's size — so training needs less memory and the
saved result is tens of megabytes.

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

<br><br>

4. Look at **STEP 2**. It lists the layers where LoRA can attach adapters. The
names differ between kinds of model, so always use the ones the model itself
reports.

<br><br>

5. Look at **STEP 3 and STEP 4**. Two settings worth knowing: **`r`** (rank)
is the adapter's size — bigger learns more but trains slower; 8-16 suits a new
format, 32-64 new behavior. **`lora_alpha`** is how strongly the adapter counts,
usually twice `r`. Then find the line STEP 4 prints ending in `(2.3437%)` or
close to it: how much of the model will be trained. Lab 2 trained 100%.

![Lab 3 trainable parameter percentage](./images/ft-lab3-trainable.png?raw=true "LoRA trains about 2% of the parameters")

<br><br>

6. Look at **STEP 5**, the other big change from Lab 2: the model is graded
only on its answer. Every position that belongs to the ticket is marked `-100`,
"don't grade this":

```
labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
```

The script prints something like `53 tokens, 36 from the prompt (not graded),
17 graded`. Lab 2 also graded the model on repeating the ticket back.

<br><br>

7. Watch the training loss fall. This lab uses a bigger learning rate (step
size) than Lab 2, `1e-3` instead of `3e-4`, which is safe because only the
small adapter is changing.

<br><br>

8. Look at **STEP 7**. The saved adapter is a few tens of MB; a full copy of the
model would be about 1.4 GB. One shared model, plus a small adapter per task or
customer.

<br><br>

9. Look at **STEP 8** and compare the scores with Lab 2's — category accuracy
should be clearly better. (The model changed too, so this shows the workshop's
progress, not how much LoRA alone helped.)

![Lab 3 scores](./images/ft-lab3-scores.png?raw=true "Lab 3 scores after LoRA training")

<br><br>

10. **Optional — change one thing and rerun:**

- **Smaller adapter (about 4 minutes).** In STEP 3 of your file, change `r=16`
  to `r=8` and `lora_alpha=32` to `16`. The trained percentage should halve;
  STEP 8 shows whether the scores suffer.
- **DoRA (about 10 minutes — best done at home).** Add `use_dora=True,` to the
  LoRA settings in STEP 3. Often a little better with small adapters, but
  noticeably slower.

Lab 5 uses the adapter this lab saves, so put the settings back and rerun
before you move on.

<br><br>

11. **Discussion — is LoRA always the better choice?** For a new format, style
or task, usually yes, and far cheaper. Retraining the whole model can still win
when the model must absorb a lot of new knowledge.

<p align="center">**[END OF LAB]**</p></br></br>

---

## **Lab 4 - QLoRA: 4-bit Quantization, Merging, and Export**

**Purpose: Store the model in a compressed 4-bit form to save memory, show that
an adapter can still be trained on top of it, measure what that costs on a CPU,
and combine the adapter and the model into one file you can ship.**

**Estimated time: 10-12 minutes**

<br><br>

**The idea in plain words.** LoRA trains only a small adapter, but the whole
model still sits in memory. QLoRA stores each of the model's numbers in 4 bits
instead of 16 or 32 — like saving a photo at lower quality — then trains a LoRA
adapter on top, to fit a bigger model into the memory you have.

**Before you run it:** this lab shows *how* QLoRA works and what it saves, not
how good its results are. It trains for only **4 steps**: on a CPU, 4-bit math
is 20-30 times slower per example than Lab 3. **QLoRA saves memory, not time.**

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
  at 16 bits (2 bytes each):      690 MB
  at 32 bits (4 bytes each):    1,380 MB   (what Lab 3 used)
  ...
  at 4 bits (0.5 byte each):      240 MB
```

Not a full 8 times smaller than 32-bit: some parts of the model stay
uncompressed. The bigger the model, the closer you get.

<br><br>

5. Look at **STEP 3**, inside one compressed layer: two 4-bit numbers per byte,
compressed in groups of 64, each group with its own scale. `nf4` is the 4-bit
format QLoRA uses.

![Lab 4 memory comparison and a look inside a 4-bit layer](./images/ft-lab4-memory.png?raw=true "4-bit memory use")

<br><br>

6. Look at **STEP 5**. The loss falls, so training still reaches the adapter.
Then read the timing:

```
  4-bit, 2 examples per step :  ~45-50 seconds per step
  Lab 3, 8 examples per step :   ~6-10 seconds per step
```

Lab 3 also handled four times as many examples per step, so per example 4-bit
is 20-30 times slower here. **Takeaway:** use QLoRA when memory is the problem;
if the model already fits, plain LoRA is faster.

<br><br>

7. Look at **STEP 6**, the merge: folding the adapter permanently into the
model, so you ship one ordinary model. We trained on the 4-bit model but merge
into a fresh full-size (16-bit) copy; merging into the compressed copy would
make its accuracy loss permanent.

<br><br>

8. Merge, or keep the adapter separate? A merged model needs nothing extra to
run, but you lose Lab 3's one-model, many-adapters trick. Merge for one
dedicated use; keep adapters separate for many customers or tasks.

<br><br>

9. Look at **STEP 7**, which scores the merged model. Poor, as expected after
4 training steps: that shows how much the amount of training matters, not that
QLoRA is worse than LoRA.

**Homework:** a fair test (24 steps, same settings) takes about 70 minutes on a
CPU. Set `QLORA_STEPS = 24` and `QLORA_BATCH = 8`, run it, and compare with
Lab 3's row in `out/results.json`. Published research found almost no
difference in quality.

<br><br>

10. Look at **STEP 8**. TinyLlama, a 1.1-billion-parameter model, loaded in
4-bit takes less memory than Lab 3's 360-million-parameter model did at full
size. On the same hardware, you can fine-tune a model about three times bigger.

<br><br>

11. **Optional, at home, on a GPU.** Point `BASE_MODEL` at TinyLlama and nothing
else needs to change. For `"microsoft/Phi-3-mini-4k-instruct"` (3.8 billion
parameters, a 7.6 GB download), also change `target_modules` to
`["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]`: Phi-3 names its layers
differently (see Lab 3, STEP 2).

<p align="center">**[END OF LAB]**</p></br></br>

---

## **Lab 5 - Systematic Evaluation**

**Purpose: Put all four runs side by side, see which kinds of tickets the model
gets wrong, and check whether fine-tuning damaged the model's general
abilities.**

**Estimated time: 10-12 minutes**

<br><br>

1. First, quiz the original SmolLM2 model on standard general-knowledge
questions with EleutherAI's lm-evaluation-harness, the standard tool for this.
About 1-2 minutes:

```
lm_eval run --model hf \
  --model_args pretrained=HuggingFaceTB/SmolLM2-360M-Instruct,dtype=float32 \
  --tasks arc_easy,arc_challenge \
  --device cpu \
  --limit 20 \
  --batch_size 1 \
  --output_path out/harness-base
```

It prints a lot of technical detail and some warnings; ignore them. The Lab 5
script reads its results file.

<br><br>

2. Now the same quiz **with your Lab 3 adapter attached** (the `peft=` part).
This needs `out/smollm2-lora` from Lab 3. Another 1-2 minutes:

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
left — and save. The last `# TODO` is for the optional Groq step; merge it
whether or not you have a key:

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
run                                      format  category  priority   queue      MB  steps    sec
-------------------------------------------------------------------------------------------------
Lab 1  bare prompt distilgpt2  82M        0.0%     0.0%     0.0%   0.0%       -      -      -
Lab 1  best prompt SmolLM2  360M        100.0%    30.0%    45.0%  45.0%       -      -      -
Lab 2  retrain all distilgpt2  82M        ...      ...      ...    ...      ...     72    ...
Lab 3  LoRA        SmolLM2  360M          ...      ...      ...    ...      ...     24    ...
Lab 4  QLoRA test  SmolLM2  360M          ...      ...      ...    ...      ...      4    ...
  (random guessing, on average: category 12.5%, priority 25.0%, queue 16.7%)
```

Read every column: `format` (right shape?), `category` (understood the
ticket?), `MB` (how big is what you would ship?), `steps` and `sec` (how much
training, how long). Compare every trained row with the **best prompt** row,
the score to beat. The Lab 4 row trained for only 4 steps, so its low scores
show too little training — not what 4-bit costs.

![Lab 5 comparison table](./images/ft-lab5-table.png?raw=true "Comparison table across every run")

<br><br>

7. Look at **STEP 3**: each category scored separately, plus the most common
mix-ups (true category → predicted). An overall score says *whether* there is a
problem; this says *where*. The usual fix for a weak category is more training
examples of it.

![Lab 5 scores by category](./images/ft-lab5-categories.png?raw=true "Scores for each category, and the most common mix-ups")

<br><br>

8. Look at **STEP 4**, four actual failures. They show in seconds whether the
model is confused or your labels are unclear — different problems, different
fixes.

<br><br>

9. Look at **STEP 5**, the general-knowledge scores before and after training:

```
  quiz                 before   +adapter     change
  --------------------------------------------------
  arc_challenge         0.250      0.250     +0.000
  arc_easy              0.250      0.300     +0.050
```

Expect changes near zero — with 20 questions, one answer moves a score by 0.05,
so a small change is noise. Watch for a **big drop**, called *catastrophic
forgetting*: the model got worse at everything except your task. The fix is
gentler training, or mixing general examples into the training data.

<br><br>

10. Two honesty notes. 20 questions per quiz catches a big drop but is not a
reliable score; remove `--limit 20` for a real one (much slower). And these
labs reuse Lab 1's test tickets throughout; in real projects, keep a separate
set for tuning decisions and use the test set once, at the end.

<br><br>

11. **Optional — compare with a big hosted model.** **STEP 6** sends the same 20
tickets to `openai/gpt-oss-120b`, a very large model hosted by Groq, with the
best prompt we could write, and compares it with your LoRA model.

This needs a free Groq API key: follow "Optional: a free Groq API key" in
`README.md` (about two minutes), then run `python finetune/lab5_eval.py` again.
**Without a key, the step prints a note and skips.** (If it reports a "404",
Groq has retired the model; the printout says what to do.)

Whatever the result: a hosted model costs per request, sends every ticket
off-site, and can be retired. And if it wins, the task may not need fine-tuning
at all. **Run this comparison before you invest in fine-tuning, not after.**

<br><br>

12. **Discussion — the five questions you can now answer with evidence.**

| Question | Where the answer is |
|---|---|
| Does fine-tuning help this task at all? | Lab 1 best-prompt row vs. the trained rows |
| Do we need to retrain the *whole* model? | Lab 2 vs. Lab 3: scores, MB, seconds |
| Do we need a bigger model? | category accuracy by model size |
| Did we break anything? | STEP 5 |
| Should we have just used a hosted model? | STEP 6 |

In one or two sentences, write the recommendation you would send Northwind
Logistics: which model (if any) goes to production, and the two numbers from
today that defend it.

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
| **Step** | One round of adjusting the model's numbers (after one batch). |
| **Epoch** | One full pass over the training data. 281 examples at batch 8 ≈ 35 steps per epoch. |
| **Token** | The unit models read — roughly a word fragment. Our tickets are ~50-70 tokens. |
| **Tokenizer** | The tool that turns text into tokens (and back), and adds a chat model's special markers. |
| **End-of-text / padding token** | The end-of-text token is the stop signal the model must learn. The padding token is filler that makes a group of examples the same length. They must be *different* tokens — see Lab 2. |
| **Masking / -100** | Marking positions with -100 so they aren't graded — used so the model is graded only on its answer, not on repeating the ticket. |
| **Base vs. instruct model** | A base model only continues text. An instruct model had a further training pass to follow instructions. distilgpt2 is base; SmolLM2-360M-Instruct is instruct. |
| **Rank (r)** | The size of a LoRA adapter. Bigger can learn more, but is bigger and slower to train. |
| **Alpha (lora_alpha)** | How strongly the adapter's adjustments count. The usual setting is twice r. |
| **Adapter** | The small add-on LoRA trains (tens of MB), saved separately from the unchanged model. |
| **Quantization / nf4** | Storing a model's numbers in fewer bits to save memory. "nf4" is the 4-bit format QLoRA uses. |
| **Merge** | Folding an adapter into the model itself to get one ordinary model. Always merge into a full-size (16- or 32-bit) copy. |
| **Test set** | Tickets the model never trained on. The only scores that count. |
| **Baseline** | The best score you can get without training — what every later run is measured against. Ours is the Lab 1 best-prompt score. |
| **Catastrophic forgetting** | Training hard on a narrow task degrading everything else the model could do. Lab 5's general-knowledge quiz (STEP 5) is the check. |

<br><br>

---

<p align="center">
<b>For educational use only by the attendees of our workshops.</b>
</p>
<p align="center">
<b>(c) 2026 Tech Skills Transformations and Brent C. Laster. All rights reserved.</b>
</p>
