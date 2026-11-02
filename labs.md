# Fine-Tuning Small AI Models: LoRA, QLoRA & Evaluation

## Revision 1.5 - 10/29/26

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

![Hover notes in the diff view](./images/merge-info3.png?raw=true "Hover over merge blocks to see what the code does")

**Screenshots.** Image placeholders in this document are being captured against
a live Codespace run and will land in `images/` before delivery.

**New to training terminology?** There is a one-page glossary at the end of this
document — loss, gradients, learning rate, batch, step, epoch, and friends. Keep
it open in a second tab if the vocabulary is new to you.

<br><br>

---

## **Lab 1 - Preparing a Dataset and Measuring a Baseline**

**Purpose: Turn raw support tickets into a properly formatted Hugging Face
dataset, see what a chat template does to it, and measure how badly an
untrained model performs on the task. Everything after this lab is measured
against the number you produce here.**

**Estimated time: 10-12 minutes**

<br><br>

**Our running task for the whole workshop.** Northwind Logistics wants every
inbound support ticket turned into one strict line:

```
category: access-management | priority: P2 | queue: identity-team
```

Eight categories, four priorities, six queues. This is a *proprietary
classification schema* — it exists nowhere in any model's training data. That
is exactly the case where fine-tuning beats prompt engineering.

<br><br>

1. Open a terminal in your Codespace (`Ctrl+Shift+` `` ` ``) and make sure you
are at the repository root. Then look at the raw data:

```
head -3 data/tickets_train.jsonl
wc -l data/tickets_train.jsonl data/tickets_eval.jsonl
```

You should see 281 training rows and 71 evaluation rows, each with a `ticket`
field and a `label` field.

**A note on the split.** Today we keep two splits: *train* and a *held-out eval*
set the model never trains on. In production you keep **three** — train,
*validation* (where every tuning decision gets made), and *test* (touched once,
at the end). And with synthetic or templated data like this, you also check that
near-duplicate tickets have not leaked across splits before believing any score.
Two splits keeps the workshop simple; Lab 5 returns to the three-way discipline.

<br><br>

2. Open the Lab 1 skeleton file so you can see what you are about to build:

```
code finetune/lab1_prep.py
```

Scroll through it. There are four `# TODO` markers. Note that this file is
incomplete — it will not produce useful output until you merge in the
completed code in the next step.

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

This takes about 25-40 seconds, most of which is the baseline generation in
Step 7. You may see warnings about `loss_type` or about pad tokens — these are
harmless.

<br><br>

5. Look at the **STEP 1** and **STEP 2** output. The `DatasetDict` printout is
what `datasets` gives you: named splits, a column list, and a row count. The
label distribution histogram matters more than it looks — a dataset where one
category has 80% of the rows will produce a model that predicts that category
for everything.

<br><br>

6. Look at **STEP 3**. This is the format the trainer actually consumes:

```
### Ticket:
Hi team, I cannot sign in to DriverPortal. It says my account is locked...

### Triage:
```

...and separately, the completion the model must produce. Splitting it into two
columns is not cosmetic — in Lab 3 you will see TRL use that split to mask the
prompt out of the loss automatically.

<br><br>

7. Look at **STEP 4**, the chat-template rendering. Same content, wrapped in
`<|im_start|>` / `<|im_end|>` markers. Every instruction-tuned model family has
its own template baked into its tokenizer. Two things worth internalizing:

- You get the right template for free by calling `tokenizer.apply_chat_template()`
- Hand-rolling the markers, or borrowing another family's, degrades quality
  silently — no error, just worse results

<br><br>

8. Look at **STEP 5**. Median sequence length should be around 50-70 tokens.
This is the single biggest reason this workshop runs on a CPU at all.
Attention cost grows quadratically with sequence length; a 4x longer dataset
would be far more than 4x slower.

<br><br>

9. Now the important part — **STEP 7 and STEP 8**. Read the three sample
predictions. distilgpt2 has never seen this schema, so it does not produce
anything close to it. Your scores should be:

```
  === BASELINE - distilgpt2, no training (n=20) ===
    schema valid :   0.0%
    category acc :   0.0%
    priority acc :   0.0%
    queue acc    :   0.0%
```

![Lab 1 baseline scores](./images/ft-lab1-baseline.png?raw=true "Baseline scores are zero across the board")

Why so *completely* zero? Partly size, partly **kind of model**. distilgpt2 is a
**base** model: trained only to continue text, never instruction-tuned, so it
does not even try to "answer" — it rambles onward from wherever the prompt
stopped. The SmolLM2-360M-**Instruct** model you meet in Lab 3 has had an
instruction-tuning pass on top of pretraining. That is a second difference —
besides size and training — to keep in mind whenever you compare rows across
labs.

<br><br>

10. **Discussion — why not just prompt-engineer this?** You could put the
eight categories, four priorities, and six queues into the prompt and ask
nicely. On a large frontier model, that works. On an 82M model there is not
enough capability to follow a multi-part instruction reliably, and on *any*
model that prompt has to be re-sent, re-tokenized, and re-paid-for on every
single request. Fine-tuning moves that knowledge into the weights: shorter
prompts, lower latency, lower cost per call, and — crucially — a model small
enough to run where you need it.

Keep the baseline number in mind. Everything from here is measured against zero.

<br><br>

11. **Optional (1 minute to change, ~1 extra minute to run) — feel the sample
size.** Change `N_EVAL = 20` near the top of `finetune/lab1_prep.py` to `40` and
rerun. The baseline is still zero, but the run takes roughly twice as long —
and that is the whole trade: every evaluation example buys confidence and costs
time. Lab 5 hits the same trade from the other side with the harness's
`--limit` flag. Put it back to `20` before you move on.

<p align="center">**[END OF LAB]**</p></br></br>

---

## **Lab 2 - Full Fine-Tuning with the Hugging Face Trainer**

**Purpose: Fine-tune every parameter of distilgpt2 with the standard Hugging
Face `Trainer`, and measure what that costs in time and disk. This is the
baseline that LoRA and QLoRA are compared against.**

**Estimated time: 10-12 minutes**

<br><br>

1. Open the Lab 2 skeleton and read through it before merging:

```
code finetune/lab2_fullft.py
```

There are four `# TODO` markers. Again — this file is incomplete and will not
train correctly until you merge.

<br><br>

2. Merge in the completed code — work top to bottom until no highlighting is
left — and save:

```
code -d extra/lab2_fullft_complete.txt finetune/lab2_fullft.py
```

<br><br>

3. Run it. **Training takes about 2-4 minutes on 4 cores** — start it now and
read the notes below while it runs:

```
python finetune/lab2_fullft.py
```

<br><br>

4. While that runs, look at the **STEP 1** output. Every one of distilgpt2's
82 million parameters is trainable — `100.0000%`. (124M is GPT-2 *small*;
DistilGPT2 is its 6-layer distillation at 81.9M.) Write that number down — in
Lab 3, PEFT's equivalent line reads about **2.3%**.

<br><br>

5. Read the `<|pad|>` note printed in **STEP 1**. This is a real bug that ships
in a lot of tutorials. GPT-2 has no padding token, so the reflex is
`tokenizer.pad_token = tokenizer.eos_token`. But the collator sets every PAD
position in the labels to `-100` so padding does not affect the loss — and if
PAD and EOS are the same token id, that erases every *real* EOS too. The model
then never learns where to stop and generates until it hits the token limit.
Two distinct tokens, one class of bug avoided.

<br><br>

6. Look at **STEP 3**. Compare the `input_ids` tail against the `labels` tail:

```
  input_ids tail : [..., 50257, 50257, 50257]
  labels tail    : [...,  -100,  -100,  -100]
```

`-100` is PyTorch's "ignore this position" label. The EOS token id survives in
the labels; the padding does not.

<br><br>

7. Watch the training loss in the progress output. It should fall steadily
from roughly 3-4 down toward 1 or below. A loss that is flat, or that jumps
around wildly, means the learning rate is wrong — that is the first thing to
check when a fine-tune misbehaves.

One honesty note while you watch it fall: a falling **training** loss only
proves the model is fitting the training set. Whether that transfers to tickets
it has never seen is what the held-out scores in STEP 7 are for — never report
a loss curve as evidence of quality.

![Lab 2 training loss falling](./images/ft-lab2-training.png?raw=true "Training loss during full fine-tuning")

<br><br>

8. When training finishes, note two numbers from **STEP 5 and STEP 6**: the
wall-clock training time, and the size of the saved model directory (a few
hundred MB). Full fine-tuning produces a complete new copy of the model.
Ten fine-tuned variants for ten customers means ten copies.

<br><br>

9. Look at **STEP 7**. Expect a dramatic jump in `schema valid` — the model
learned the *shape* of the answer very quickly — while category accuracy
lags well behind. That gap is the lesson of this lab: **learning a format is
easy; learning the semantics behind the format needs more model.**

<br><br>

10. **Discussion — when would you actually do this?** Full fine-tuning is the
right call when you have a genuinely large task-specific dataset (tens of
thousands of examples or more), when you need the model to acquire new
capability rather than a new output style, and when you can afford the compute
and the per-variant storage. For everything else — which is most enterprise
work — the next two labs are the better trade.

<br><br>

11. **Optional (~2 min of training) — halve the supervision.** Change
`num_train_epochs=2` to `1` in `finetune/lab2_fullft.py`, rerun, and compare
the STEP 7 scores against your first run. Then put it back. Notice what you
just did: changed **one** variable and held everything else still — your first
controlled experiment of the afternoon, a pattern Lab 5 comes back to.

<p align="center">**[END OF LAB]**</p></br></br>

---

## **Lab 3 - LoRA Fine-Tuning with PEFT**

**Purpose: Fine-tune a model four times larger than Lab 2's while training
about 2.3% of its parameters, using PEFT's `LoraConfig`. Mask the prompt out of
the loss. Produce an adapter measured in megabytes instead of hundreds of
megabytes.**

**Estimated time: 10-12 minutes**

<br><br>

**The idea in one line.** LoRA freezes the base weights and learns a low-rank
correction beside them:

```
    h = W0 @ x  +  (alpha / r) * B @ A @ x
        ^frozen         ^trainable, rank r
```

`A` is `r x in_features`, `B` is `out_features x r`. Because `r` is small
(8, 16, 32) next to layer widths in the hundreds or thousands, `B @ A` costs a
few million parameters instead of a few hundred million — while still being
able to express a full-size update to the layer.

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

3. Run it. **Budget about 5 minutes total** — roughly 1 minute to load the
fp32 model, 3 minutes of training, and a minute of evaluation:

```
python finetune/lab3_lora.py
```

You may see a line mentioning `gemm_4bit_forward` and `kernels` — it is a
harmless message from the 4-bit library (it appears in Labs 3-5). Ignore it.

<br><br>

4. Look at **STEP 2**, the list of `Linear` module names. This is
architecture-specific and it is the most common way LoRA configs go wrong:

| Family | Attention modules | MLP modules |
|---|---|---|
| Llama-style (SmolLM2, TinyLlama, Qwen, Mistral, Llama 3) | `q_proj` `k_proj` `v_proj` `o_proj` | `gate_proj` `up_proj` `down_proj` |
| Fused-projection (Phi-3, Phi-4-mini) | `qkv_proj` `o_proj` | `gate_up_proj` `down_proj` |
| GPT-2 style (distilgpt2) | `c_attn` `c_proj` | `c_fc` `c_proj` |

Three things to take from that table. Phi-3 packs Q, K and V into a single
`qkv_proj` tensor, so the Llama names match *nothing* on it — a trap, because
Phi-3 is otherwise Llama-like. GPT-2's layers are `Conv1D`, not `Linear`; PEFT
handles them, but again under different names.

And the failure mode changed recently, in a good way: **since PEFT 0.21 a
`target_modules` list that matches nothing raises `NoMatchingPeftModuleError`**
instead of silently adapting zero layers while training runs happily to
completion. A list that matches only *some* of its names still succeeds
quietly, though, so sanity-check the trainable count rather than assuming.
When in doubt, `target_modules="all-linear"` lets PEFT work it out.

<br><br>

5. Look at **STEP 3 and STEP 4**. Two dials worth understanding:

- **`r` (rank)** — capacity. Higher `r` can express a richer update and costs
  proportionally more parameters. 8-16 is plenty for style and format
  adaptation; 32-64 for tasks that need new behavior.
- **`lora_alpha`** — scaling. The update is multiplied by `alpha / r`. The
  common convention `alpha = 2 * r` keeps the effective strength constant as
  you change rank, so you can tune `r` without re-tuning your learning rate.

Then read PEFT's `print_trainable_parameters()` line — it reports
`trainable%: 2.34` or thereabouts. Compare against the `100.0000%` that Lab 2
printed. (Note Step 1 of this lab also prints `100.0000%`, because that is the
base model *before* LoRA is applied.)

![Lab 3 trainable parameter percentage](./images/ft-lab3-trainable.png?raw=true "LoRA trains about 2% of the parameters")

<br><br>

6. Look at **STEP 5**, the masking. This is the one real difference from
Lab 2. For every example we tokenize the prompt and the full string separately,
then overwrite the prompt positions in `labels` with `-100`:

```
labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
```

The script prints how many tokens were masked versus graded — something like
`53 tokens, 36 masked as prompt, 17 graded`. In Lab 2 we did not do this, so
distilgpt2 spent part of its capacity learning to re-predict ticket text we
already had. On a small model that difference is real.

`DataCollatorForSeq2Seq` is what makes it work at batch level: it pads
`input_ids` with the pad token and `labels` with `-100`, which is exactly what
completion-only training needs.

> **A note on TRL.** TRL's `SFTTrainer` does this masking for you automatically
> when your dataset columns are named `prompt` and `completion`, and on a GPU it
> is what you would reach for. We use the plain Hugging Face `Trainer` here for
> two reasons: you get to see exactly what the masking means, and on a CPU-only
> Codespace `SFTTrainer` measured roughly **31x slower** than `Trainer` for the
> identical model, shapes and LoRA config — minutes versus over an hour.

<br><br>

7. Watch the training loss. Because only the adapters move, LoRA tolerates a
much higher learning rate than full fine-tuning. This lab uses `1e-3` against
Lab 2's `3e-4` — and Lab 2's own rate is already pushed high because the run is
short. For a full fine-tune of any real length you would be at `1e-5` to `5e-5`,
so `1e-3` is 20-100x what full fine-tuning would tolerate.

<br><br>

8. Look at **STEP 7**, the artifact size comparison. The adapter should be a
few tens of MB against roughly 1.4 GB for a full copy. This is the
deployment story: **one base model in memory, many adapters on disk.** Serving
stacks like vLLM and TGI can hold one base model and hot-swap adapters per
request, which is how a single GPU serves twenty fine-tuned variants.

<br><br>

9. Look at **STEP 8**. Compare against Lab 2: a 4x larger model, about 2.3% of
the parameters trained, an artifact roughly 40x smaller than a full copy — and
category accuracy that should be clearly better.

One caution before you quote that comparison anywhere: between Lab 2 and Lab 3
the model, its size, its instruction tuning, the training method, the step
count, and the loss masking **all changed at once**. These rows show the
workshop's progression; they do not isolate LoRA versus full fine-tuning. Fair
comparisons change one variable at a time — the nearest thing to one in this
workshop is Lab 3 versus Lab 4's take-home run: same model, data, rank, and
step budget, with only the base precision changed.

<br><br>

10. **Optional (about 2 extra minutes) — try DoRA.** Set `use_dora=True` in
Step 3 of your file and rerun. DoRA (Weight-Decomposed Low-Rank Adaptation)
splits each weight into a magnitude and a direction, and applies the low-rank
update only to the direction. It generally recovers more of full fine-tuning's
quality at low rank, at some cost in speed. At `r=16` the difference is small;
the real use case is `r=4` or `r=8`, where plain LoRA starts to run out of
capacity.

**Or turn the other dial — rank.** Change `r=16` to `r=8` (and `lora_alpha=32`
to `16`, keeping the `alpha = 2r` convention) and rerun. The
`print_trainable_parameters()` line and the adapter size in STEP 7 should both
roughly halve; the STEP 8 scores tell you whether this task actually needed the
extra capacity. One variable, everything else held still.

<br><br>

11. **Discussion — so is LoRA just strictly better?** Nearly, for adaptation
tasks. The honest caveats: LoRA adds a small inference cost unless you merge
(Lab 4); it is weaker than full fine-tuning when the model must absorb genuinely
new knowledge rather than new behavior; and `r`, `alpha`, and `target_modules`
are three more things to get wrong. The 90-95%-of-full-quality figure from the
LoRA literature is for *adaptation*, which is what most enterprise work is.

<p align="center">**[END OF LAB]**</p></br></br>

---

## **Lab 4 - QLoRA: 4-bit Quantization, Merging, and Export**

**Purpose: Quantize the frozen base to 4-bit NormalFloat, prove that gradients
still reach the adapters through it, measure what that costs on CPU, and merge
and export the result correctly.**

**Estimated time: 10-12 minutes**

<br><br>

**Why this lab exists.** LoRA cut the *trainable* parameters. The frozen base
model is still sitting in memory at 2 or 4 bytes per weight, and for anything
above a couple of billion parameters that is what stops you. QLoRA quantizes
the frozen base to 4-bit NormalFloat, keeps the adapters at higher precision,
and backpropagates through the quantized weights.

**Read this before you run it.** This lab is about **mechanics and memory**,
not about quality. We train for only **4 steps**, and Step 5 makes you measure
exactly why: on this CPU, bitsandbytes 4-bit matmuls run roughly **20-25x
slower per step** than float32, so a Lab 3-sized 24-step run measures over an
hour. bitsandbytes' dequantize-and-multiply path is a CUDA kernel; on CPU there
is no equivalent, so you pay the dequantization in full on every matmul.

That is not a workshop compromise — it is the actual shape of the technique.
**QLoRA buys memory, not speed.** On a GPU under VRAM pressure that trade is
what makes the run possible at all (the QLoRA paper fine-tuned a 65B model on a
single 48 GB card), and there the overhead versus 16-bit LoRA is roughly a
third. Be straight with your team about which of the two you are buying.

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

3. Run it. **This one takes about 6-7 minutes total** — the 4 training steps
are about 3 minutes of it, and Step 8 loads a second model:

```
python finetune/lab4_qlora.py
```

<br><br>

4. Look at **STEP 1 and STEP 2**. You should see something close to:

```
  bfloat16   :     ~690 MB   (2 bytes per weight)
  float32    :   ~1,380 MB   (4 bytes per weight -- what Lab 3 used)
  4-bit NF4  :     ~240 MB
```

Note that this is not the 8x you would naively predict going from 32 bits to
4 bits. Embeddings and the LM head stay unquantized — and on SmolLM2 the tied
embedding table alone is 47M of the 362M parameters — plus every 64-weight block
carries its own scale factor. Expect roughly 2.5-3x against bfloat16 and 5-6x
against float32. (The script divides by 1024², so these are MiB, like `ls -h`.) The larger the model, the closer you get to the theoretical
ratio, which is why QLoRA's headline numbers come from 7B-70B models, not 360M
ones.

<br><br>

5. Look at **STEP 3**, the `Linear4bit` inspection. Three ideas to take away:

- **Packed storage** — the weight tensor's dtype is `uint8`, two 4-bit values
  per byte, which is why the shape looks flattened.
- **Blockwise** — `blocksize: 64`. Each block of 64 weights gets its own scale,
  so one outlier weight cannot wreck the precision of a whole tensor.
- **NormalFloat** — NF4's 16 levels are not evenly spaced. They are placed so
  each level receives an equal share of a normal distribution, which is what
  trained weights actually look like. That is why NF4 beats plain int4 at the
  same bit width.

And `bnb_4bit_use_double_quant=True` quantizes the block scales themselves,
saving roughly another 0.4 bits per parameter.

![Lab 4 memory comparison and Linear4bit inspection](./images/ft-lab4-memory.png?raw=true "4-bit NF4 memory footprint")

<br><br>

6. Look at **STEP 5**, and read the two numbers it prints. The loss moves,
which answers the question that matters: **gradients really do flow back
through packed 4-bit weights to reach the adapters.** If they did not, QLoRA
would not exist.

Then read the timing comparison. The script pulls Lab 3's float32 seconds-per-step
straight out of `out/results.json`, so you are comparing two runs *you* made on
*this* machine:

```
  4-bit NF4, batch 2       : ~43 s/step
  Lab 3 float32, batch 8   : ~8-10 s/step
```

Per step that is roughly 4-5x — but Lab 3's step processed **four times as many
examples**, so per example the gap is around **20x**. Notice also that we
dropped the batch size to 2: 4-bit CPU matmuls scale badly with batch, and at
batch 8 the same step measured about **177 seconds**.

> **Takeaway to write down:** reach for QLoRA when **memory is the binding
> constraint**. If the model already fits in the memory you have, expect plain
> LoRA to train faster — the exact gap depends on hardware and kernels (you just
> measured a CPU's version of it), but speed is not what quantization is for.

<br><br>

7. Look at **STEP 6**, the merge. This is the step people get wrong, so read
the code carefully: we trained against a **4-bit** base but merged into a
**bfloat16** base. Merging into quantized weights would bake the quantization
error permanently into your model. The correct order is always: train
quantized → merge into full precision → re-quantize the merged result if you
need to.

<br><br>

8. Note what merging buys and costs. A merged model is an ordinary Hugging
Face model directory — no PEFT dependency at runtime, no adapter-loading step,
and no per-layer adapter overhead during inference. What you give up is the
Lab 3 trick of swapping adapters against one shared base. Merge for a single
dedicated deployment; keep adapters separate for multi-tenant serving. The
export procedure is identical whether the adapter took 4 steps or 4,000.

<br><br>

9. Look at **STEP 7**. First note *what* is being scored: the merged bfloat16
model from Step 6, not the adapter stack — you evaluate the exact artifact you
would ship, because training against a quantized base leaves its fingerprints
on the adapter either way. The scores are poor, and they are *supposed* to be —
eight examples of supervision is not fine-tuning. Do not read this row as
"QLoRA is worse than LoRA." Read it as the cheapest possible demonstration
that **step count dominates every knob we have discussed all afternoon**.

The controlled quality experiment — same model, same data, same rank, 24
steps, float32 base versus 4-bit base — is a genuinely worthwhile thing to
run. It is also a ~70-minute CPU job, so it is homework: set
`QLORA_STEPS = 24` and `QLORA_BATCH = 8`, run it overnight, and compare
against Lab 3's row in `out/results.json`. The published expectation is that
you will find almost no gap; the QLoRA paper reports 4-bit NF4 with double
quantization matching 16-bit LoRA on their benchmarks. **Quantization trades
memory for speed, not memory for accuracy.**

<br><br>

10. Look at **STEP 8**. TinyLlama at 1.1B parameters, loaded in 4-bit, occupies
less memory than the 360M model did at float32 in Lab 3. That is the whole
argument in one comparison: **quantization moves the size of model you can
fine-tune on given hardware up by about one tier.** It is also the one result
from this lab that transfers unchanged to your real hardware.

<br><br>

11. **Optional, at home, on a GPU.** Point `BASE_MODEL` at TinyLlama and every
other line of the script stays identical — it is Llama-style, so the
`target_modules` list already matches. That is the real claim QLoRA makes: not
that it is fast, but that the code barely changes as the model grows.

Going further, to `"microsoft/Phi-3-mini-4k-instruct"` (3.8B), needs exactly
one more edit — `target_modules` becomes
`["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]`, because Phi-3 fuses its
projections. Leave the Llama names in and PEFT will raise
`NoMatchingPeftModuleError`, which is Lab 3 Step 2's lesson arriving to collect.
It is also a 7.6 GB download with multi-minute CPU steps, which is why it is
homework and not a lab.

<p align="center">**[END OF LAB]**</p></br></br>

---


## **Lab 5 - Systematic Evaluation**

**Purpose: Put all four runs in one table, break the errors down by class,
and use EleutherAI's lm-evaluation-harness to check whether fine-tuning damaged
the model's general capability.**

**Estimated time: 10-12 minutes**

<br><br>

1. Run the harness against the **base** model first. This writes a JSON result
file that the Lab 5 script will read. It takes about 1-2 minutes:

```
lm_eval run --model hf \
  --model_args pretrained=HuggingFaceTB/SmolLM2-360M-Instruct,dtype=float32 \
  --tasks arc_easy,arc_challenge \
  --device cpu \
  --limit 20 \
  --batch_size 1 \
  --output_path out/harness-base
```

Both tasks come from the same public `allenai/ai2_arc` dataset, so this is one
download. You may see warnings about the number of few-shot examples — harmless.
If you want a broader signal, `piqa`, `sciq`, and `hellaswag` all work too; keep
`--limit` low on `hellaswag`, which is much heavier.

<br><br>

2. Now run the same evaluation **with your Lab 3 adapter attached**. This step
and Steps 4-8 need `out/smollm2-lora`, which Lab 3 produced — if Lab 3 did not
finish, go back and complete it first. Note the
`peft=` argument — that is how the harness loads a LoRA adapter on top of a
base model. Another 1-2 minutes:

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
left — and save. There are four `# TODO` markers here; the last one belongs to
the optional Groq step and is harmless to merge whether or not you have a key:

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
Lab 2  full FT     distilgpt2  82M        ...      ...      ...    ...      ...     70    ...
Lab 3  LoRA fp32   SmolLM2  360M          ...      ...      ...    ...      ...     24    ...
Lab 4  QLoRA probe SmolLM2  360M          ...      ...      ...    ...      ...      4    ...
```

Read the columns, not just the last row. `schema` answers "can it produce the
right shape at all?"; `category` answers "does it understand the ticket?"; `MB`
answers "what do we ship per variant?"; `steps` answers "how much supervision
did it actually see?"; `sec` answers "what did it cost?".
A single "accuracy" number would have hidden every one of those trade-offs.

Read rows 3 and 4 with the `steps` column in view. Row 4 is Lab 4's **4-step
mechanics probe**, not a quality run — so this table does *not* tell you what
quantization costs in accuracy. It tells you something else worth just as much:
the gap between those two rows is almost entirely a **step-count** gap, and
undertraining is the most common reason a real fine-tune disappoints.

![Lab 5 comparison table](./images/ft-lab5-table.png?raw=true "Comparison table across all four runs")

<br><br>

7. Look at **STEP 3**, the per-category breakdown and the confusion pairs. An
aggregate score tells you *if* you have a problem; a per-class breakdown tells
you *where*. When one category drags the average down, the fix is almost always
more training examples for that category — not more epochs on the same data.

What the script prints — per-class accuracy plus confusion pairs — is the
informal version of what production evaluations formalize as per-class
precision, recall, **macro-F1**, and a full confusion matrix. Same questions,
standard names; `scikit-learn` is already installed in this environment when
you want the formal versions.

<br><br>

8. Look at **STEP 4** and actually read the failures. Metrics are lossy
summaries. Reading four wrong predictions tells you in seconds whether the
model is confused about the task or whether your labels are ambiguous — which
is a different problem with a different fix.

<br><br>

9. Look at **STEP 5**, the harness deltas. This is the question almost nobody
asks until production:

```
  task                   base   +adapter      delta
  --------------------------------------------------
  arc_easy              0.400      0.450     +0.050
  arc_challenge         0.250      0.250     +0.000
```

Your adapter made the model good at ticket triage. These numbers ask what that
cost on *everything else*. **Deltas near zero are what you should expect here**
— 24 steps of LoRA on seven projections is a very light touch, and in our runs
`arc_challenge` moved `+0.000` and `arc_easy` `+0.050`. A small *positive*
delta is noise, not improvement: at `--limit 20`, one extra correct answer
moves the number by 0.05. Do not go tell anyone that fine-tuning on support
tickets improved ARC.

The finding you ran this to catch is a **large** negative delta — that is
catastrophic forgetting, and it means your learning rate or epoch count is too
aggressive, or your dataset is so narrow the model collapsed onto one output
shape. The standard fix is to mix a slice of general instruction data into your
training set and retrain.

The point of the check is not the number, it is *having* the number. Most teams
ship a fine-tune having never measured this at all.

<br><br>

10. Be honest about `--limit 20`. Twenty items per task is a noisy estimate —
enough to catch a collapse, nowhere near enough to publish. Drop the `--limit`
flag entirely for a real number (and expect it to take considerably longer on
CPU). Knowing which of your numbers are indicative and which are conclusive is
part of the job.

And be honest about the **splits**, too. These labs reuse one held-out eval set
from Lab 1 onward — fine for a workshop, exactly what you should *not* do in
production. Every time you peek at a test set and then change something, it
quietly becomes a validation set. Keep three splits, make tuning decisions only
on validation, touch test once at the end, and check templated data for
near-duplicates across splits before believing any number.

<br><br>

11. **Optional — the prompt-engineering ceiling.** **STEP 6** runs the same 20
tickets through `openai/gpt-oss-120b` on Groq, zero-shot, with the best prompt
we can write: every category, every priority, the full routing table, and the
exact output contract. It then prints a head-to-head against your LoRA model.

This step needs a free Groq API key. If you have not set one up, follow
"Optional: a free Groq API key" in `README.md` — it takes about two minutes and
ends with `source scripts/setup-key.sh`. Then run Step 5 above again.
**If no key is set, the step prints a note and skips** — nothing else depends
on it, and the rest of the workshop needs no accounts at all.

> **If this step 404s,** the model has been retired. Groq decommissioned
> `llama-3.3-70b-versatile` — the model this workshop originally used — on
> 2026-08-16. Check [console.groq.com/docs/deprecations](https://console.groq.com/docs/deprecations)
> and change `FRONTIER_MODEL` in `common/triage.py`. That is the only edit
> needed, and the churn is itself one of the arguments below.
> `python scripts/check_groq.py` lists the models your key can reach right now.

Read the result carefully, because it is the question your management will ask.
Whichever way accuracy falls, four things do not change: you pay per token on
every request forever, you eat a network round trip per ticket, every ticket
leaves your network, and the model can be deprecated or updated underneath you
— as this very step has already demonstrated once.
And if the frontier model *does* win here, the honest read is that this task may
be prompt-able and fine-tuning may not earn its keep — which is a legitimate
finding, and one API call well spent. **Run this comparison before you invest in
fine-tuning, not after.**

<br><br>

12. **Discussion — the five questions you can now answer with evidence.**

| Question | Where the answer is |
|---|---|
| Does fine-tuning help this task at all? | Lab 1 baseline vs. any trained row |
| Do we need *full* fine-tuning? | Lab 2 vs. Lab 3, quality per MB and per second |
| Do we need a bigger model? | category accuracy across the model sizes |
| Did we break anything? | the lm-evaluation-harness deltas |
| Should we have just called an API? | the Step 6 head-to-head |

That is the entire evaluation job. Everything else is hyperparameters.

Close the loop the way you would at work: in one or two sentences, write the
recommendation you would send Northwind Logistics — which model (if any) goes
to production, and the two numbers from today you would cite to defend it. If
you cannot fill in those two numbers, you know exactly which lab to rerun.

<p align="center">**[END OF LAB]**</p></br></br>

---

## Where to go next

- **More data beats more epochs.** 281 examples is a demo. Real deployments of
  this pattern typically use 1,000-10,000 curated examples, and curation
  quality matters more than count.
- **Preference tuning.** Once supervised fine-tuning has your format right,
  DPO and GRPO (both in TRL) tune for *preferred* outputs rather than merely
  correct ones.
- **Serving adapters.** vLLM and TGI both support multi-LoRA serving — one base
  model in GPU memory, many adapters swapped per request.
- **Export formats.** A merged model can be converted to GGUF for `llama.cpp`
  or Ollama, or quantized with AWQ/GPTQ for GPU serving.
- **Run it on a GPU.** Everything in these labs is device-agnostic. Drop
  `use_cpu=True`, and the same scripts will run 50-100x faster on a single
  consumer GPU.

<br><br>

---

## Appendix — one-page glossary

Plain-language versions of every training term used today.

| Term | What it means here |
|---|---|
| **Loss** | A single number scoring how wrong the model's predictions are on a batch. Training exists to push it down. |
| **Gradient** | For each weight, which direction (and how strongly) to nudge it to reduce the loss. Computed by backpropagation. |
| **Optimizer** | The code that applies those nudges. Ours is **Adam**, which also keeps two running averages ("moments") per weight — that is why full fine-tuning costs ~16 bytes per parameter. |
| **Learning rate** | How big each nudge is. Too high: the loss jumps around. Too low: nothing happens. |
| **Batch** | How many examples are processed together for one update. |
| **Step** | One update of the weights (one batch). |
| **Epoch** | One full pass over the training data. 281 examples at batch 8 ≈ 35 steps per epoch. |
| **Token** | The unit models read — roughly a word fragment. Our tickets are ~50-70 tokens. |
| **EOS / PAD** | End-of-sequence token (the stop signal the model must learn) / filler that makes batch rows the same length. They must be *different* tokens — see Lab 2. |
| **Masking / -100** | Marking positions the loss should ignore, so the model is graded only on the completion, not on re-predicting the ticket. |
| **Base vs. instruct model** | A base model only continues text. An instruct model had a further training pass to follow instructions. distilgpt2 is base; SmolLM2-360M-Instruct is instruct. |
| **Rank (r)** | LoRA's capacity dial: the width of the low-rank update. Higher r = more expressive, more parameters. |
| **Alpha** | LoRA's strength dial; the update is scaled by alpha / r. Convention: alpha = 2r. |
| **Adapter** | The small trained LoRA weights (tens of MB), saved separately from the frozen base. |
| **Quantization / NF4** | Storing weights in fewer bits. NF4 is the 4-bit format QLoRA uses — levels spaced to match how trained weights are actually distributed. |
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
