# Fine-Tuning Small AI Models: LoRA, QLoRA & Evaluation

## Revision 1.0 - 07/31/26

**Workshop labs — 5 labs, 10-12 minutes each**

Everything in these labs runs on a **CPU-only, 4-vCPU GitHub Codespace**. No GPU,
no paid accounts, no API keys. Every model and dataset is public and free.

*(One exception: Lab 5 has a clearly-marked **optional** final step that compares
your fine-tuned model against a 70B model on Groq. It needs a free `GROQ_API_KEY`
and skips cleanly without one. Nothing else in the workshop touches it.)*

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

**Screenshots.** Image placeholders in this document are being captured against
a live Codespace run and will land in `images/` before delivery.

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

This takes about 60-90 seconds, most of which is the baseline generation in
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

<p align="center">**[END OF LAB]**</p></br></br>

---

## **Lab 3 - LoRA Fine-Tuning with TRL**

**Purpose: Fine-tune a model four times larger than Lab 2's while training
about 2.3% of its parameters, using PEFT's `LoraConfig` and TRL's
`SFTTrainer`. Produce an adapter measured in megabytes instead of hundreds of
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

3. Run it. **Training takes about 2-3 minutes**:

```
python finetune/lab3_lora.py
```

<br><br>

4. Look at **STEP 2**, the list of `Linear` module names. This is
architecture-specific and it is the most common way LoRA configs go wrong:

| Family | Attention modules | MLP modules |
|---|---|---|
| Llama-style (SmolLM2, TinyLlama, Phi-3, Qwen, Mistral) | `q_proj` `k_proj` `v_proj` `o_proj` | `gate_proj` `up_proj` `down_proj` |
| GPT-2 style (distilgpt2) | `c_attn` `c_proj` | `c_fc` `c_proj` |

GPT-2's layers are `Conv1D`, not `Linear` — PEFT handles it, but a
`target_modules` list copied from a Llama tutorial will simply match nothing.
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

6. Look at **STEP 5**. Because the columns are literally named `prompt` and
`completion`, TRL sets `completion_only_loss=True` on its own: the prompt
tokens get `-100` labels and contribute nothing to the loss. In Lab 2 we did
not do that — distilgpt2 spent part of its capacity learning to re-predict
ticket text we already had. This is a meaningful quality difference on small
models, and TRL gives it to you for free if you name the columns correctly.

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

<br><br>

10. **Optional (about 2 extra minutes) — try DoRA.** Set `use_dora=True` in
Step 3 of your file and rerun. DoRA (Weight-Decomposed Low-Rank Adaptation)
splits each weight into a magnitude and a direction, and applies the low-rank
update only to the direction. It generally recovers more of full fine-tuning's
quality at low rank, at some cost in speed. At `r=16` the difference is small;
the real use case is `r=4` or `r=8`, where plain LoRA starts to run out of
capacity.

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

**Purpose: Run Lab 3's exact recipe against a 4-bit quantized base, measure
the memory difference, then merge the adapter into a full-precision base and
export a standalone model.**

**Estimated time: 10-12 minutes**

<br><br>

**Why this lab exists.** LoRA cut the *trainable* parameters. The frozen base
model is still sitting in memory at 2 or 4 bytes per weight, and for anything
above a couple of billion parameters that is what stops you. QLoRA quantizes
the frozen base to 4-bit NormalFloat, keeps the adapters at higher precision,
and backpropagates through the quantized weights.

This lab deliberately uses the **same model, data, rank, and step count** as
Lab 3. Only the base precision changes — so any quality difference you see is
attributable to quantization and nothing else.

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

3. Run it. **This one takes about 4-5 minutes total** — 4-bit is slower per
step on CPU, and Step 8 loads a second model:

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

6. Note the honest CPU caveat printed in Step 3: **on CPU, 4-bit buys memory,
not speed.** Every matmul has to dequantize first, so steps run roughly 1.4x
slower than bf16. On a GPU under VRAM pressure the memory saving is what lets
you train at all — that is the actual point of QLoRA, and it is worth being
straight with your team about which of the two you are buying.

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
dedicated deployment; keep adapters separate for multi-tenant serving.

<br><br>

9. Look at **STEP 7** and compare against Lab 3's numbers. Same everything
except base precision. Note the script records the **adapter** size for the
Lab 5 table — the merged directory is a separate, much larger artifact and is
recorded under its own key. On real benchmarks the QLoRA paper found this gap to be
close to zero; at this scale and step count expect small differences in both
directions.

<br><br>

10. Look at **STEP 8**. TinyLlama at 1.1B parameters, loaded in 4-bit, occupies
less memory than the 360M model did at float32 in Lab 3. That is the whole
argument in one comparison: **quantization moves the size of model you can
fine-tune on given hardware up by about one tier.**

<br><br>

11. **Optional, at home.** Change `BASE_MODEL` to
`"microsoft/Phi-3-mini-4k-instruct"` (3.8B) and rerun. Every other line of the
script stays identical. It is a 7.6 GB download and each training step takes
minutes on CPU, which is exactly why it is not a live lab — but it is the same
code, and that is the point of the whole workshop.

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
run                                      schema  category  priority   queue      MB    sec
------------------------------------------------------------------------------------------
Lab 1  baseline   distilgpt2  82M          0.0%      0.0%      0.0%    0.0%       -      -
Lab 2  full FT    distilgpt2  82M         ...       ...       ...     ...       ...    ...
Lab 3  LoRA fp32  SmolLM2  360M           ...       ...       ...     ...       ...    ...
Lab 4  QLoRA nf4  SmolLM2  360M           ...       ...       ...     ...       ...    ...
```

Read the columns, not just the last row. `schema` answers "can it produce the
right shape at all?"; `category` answers "does it understand the ticket?"; `MB`
answers "what do we ship per variant?"; `sec` answers "what did it cost?".
A single "accuracy" number would have hidden every one of those trade-offs.

![Lab 5 comparison table](./images/ft-lab5-table.png?raw=true "Comparison table across all four runs")

<br><br>

7. Look at **STEP 3**, the per-category breakdown and the confusion pairs. An
aggregate score tells you *if* you have a problem; a per-class breakdown tells
you *where*. When one category drags the average down, the fix is almost always
more training examples for that category — not more epochs on the same data.

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
  arc_easy              0.377      0.362     -0.015
  arc_challenge         0.283      0.271     -0.012
```

Your adapter made the model good at ticket triage. These numbers tell you what
that cost on *everything else*. Small negative deltas are normal and usually
acceptable. A large drop means catastrophic forgetting — your learning rate or
epoch count is too aggressive, or your dataset is too narrow. The standard fix
is to mix a slice of general instruction data into your training set and
retrain.

<br><br>

10. Be honest about `--limit 20`. Twenty items per task is a noisy estimate —
enough to catch a collapse, nowhere near enough to publish. Drop the `--limit`
flag entirely for a real number (and expect it to take considerably longer on
CPU). Knowing which of your numbers are indicative and which are conclusive is
part of the job.

<br><br>

11. **Optional — the prompt-engineering ceiling.** **STEP 6** runs the same 20
tickets through `llama-3.3-70b-versatile` on Groq, zero-shot, with the best
prompt we can write: every category, every priority, the full routing table,
and the exact output contract. It then prints a head-to-head against your LoRA
model.

This step needs a free `GROQ_API_KEY` set as a Codespaces secret. **If it is not
set, the step prints a note and skips** — nothing else depends on it, and the
rest of the workshop needs no accounts at all.

Read the result carefully, because it is the question your management will ask.
Whichever way accuracy falls, four things do not change: you pay per token on
every request forever, you eat a network round trip per ticket, every ticket
leaves your network, and the model can be deprecated or updated underneath you.
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

<p align="center">
<b>For educational use only by the attendees of our workshops.</b>
</p>
<p align="center">
<b>(c) 2026 Tech Skills Transformations and Brent C. Laster. All rights reserved.</b>
</p>
