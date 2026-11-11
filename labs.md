# Fine-Tuning Small AI Models: LoRA, QLoRA & Evaluation

## Revision 1.15 - 11/11/26

**Workshop labs**

Five labs, about 10 minutes each, on one task: turning a support ticket into
one strict line.

**Merging code.** Each lab opens a side-by-side view: the finished code on the
left, your file on the right. Click the arrow (`→`) beside each highlighted
block to copy it across, top to bottom, until nothing is highlighted. Save
(`Cmd+S` on Mac, `Ctrl+S` elsewhere) and close the tab. Hover over a block to
read what it does.

![Hover notes in the diff view](./images/merge-info3.png?raw=true "Hover over merge blocks to see what the code does")

**New words?** A one-page glossary is at the end of this document.

<br><br>

---

## **Lab 1 - The Task and the Score to Beat**

**Purpose: Measure the best an untrained model can do. Every later lab has to
beat it.**

**Time: 10 minutes**

<br><br>

**The task.** Northwind Logistics wants each support ticket turned into one line:

```
category: access-management | priority: P2 | queue: identity-team
```

8 categories, 4 priorities, 6 queues: a labeling scheme no model has seen.

<br><br>

1. In a terminal at `/workspaces/fine-tuning`, open your starter file next to
the finished one, and merge the three highlighted blocks: how a model is
asked a question, how the best prompt is built, and how answers are scored.

```
code -d extra/lab1_prep_complete.txt finetune/lab1_prep.py
```

<br><br>

2. Run it (about 2 minutes):

```
python finetune/lab1_prep.py
```

It tries two untrained models: distilgpt2 with just the ticket, then SmolLM2
(the model Labs 3 and 4 train) with the best prompt we could write. It also
opens two files in editor tabs. **Read them while it runs.** (Click back in
the terminal before you type your next command.)

- **out/lab1_best_prompt.txt**: the whole best prompt. Find the rules, the list
of labels, and the 8 worked examples written as a conversation. Notice how
long it is: it is sent with every ticket.
- **out/lab1_answers.txt**: fills in as each model answers. What does
distilgpt2 write instead of a label? Which part does the best prompt get
wrong most often?

![Lab 1 answers file](./images/ft-lab1-answers.png?raw=true "The answers file filling in while the lab runs")

<br><br>

3. Read **RESULTS**:

![Lab 1 results](./images/ft-lab1-results.png?raw=true "Random guessing, just the ticket, and the best prompt")

- **just ticket: 0%.** distilgpt2 only continues text; it never names a category.
- **best prompt** gets the format right, but still misses most categories.
**This column is the score every later lab has to beat.**
- **priority 45%** looks good, but the model answers P3 for almost every
ticket, and 9 of the 20 test tickets are P3.

<br><br>

4. **Think about it:** the best prompt is long, and you pay for it on every
request. Training puts that knowledge into the model instead, if it can beat
this score.

<br><br>

**Optional (outside the 10 minutes):** set `N_EVAL = 40` in
`finetune/lab1_prep.py` and rerun for a more reliable score. Set it back to
`20` and rerun before Lab 2.

<p align="center">**[END OF LAB]**</p></br></br>

---

## **Lab 2 - Full Fine-Tuning: Retrain the Whole Model**

**Purpose: Retrain every number in a small model, and see what it costs in
time and disk space.**

**Time: 10 minutes**

<br><br>

1. Merge every highlighted block:

```
code -d extra/lab2_fullft_complete.txt finetune/lab2_fullft.py
```

<br><br>

2. Run it (about 3 minutes):

```
python finetune/lab2_fullft.py
```

While it runs: **STEP 1** says **100%** of the model will be trained. Then
watch the **loss**, how wrong the model's guesses are, fall as it trains.
Under each loss line is the model's answer to one test ticket at that moment.
Before training it just writes more ticket-like text. Watch it learn the
format, then the labels. Which part keeps changing?

<br><br>

3. Read the line after training: how long it took, and the size of the saved
model. Retraining everything saves a whole new copy, one per customer or task.

<br><br>

4. Read **RESULTS**:

![Lab 2 results](./images/ft-lab2-results.png?raw=true "Lab 2 next to the best prompt")

A model 4 times smaller, after a few minutes of training, beats the best
prompt on category and queue. Priority stays at 45%: it still answers P3 for
almost everything.

<br><br>

**Optional (outside the 10 minutes):** change `num_train_epochs=2` to `1`
(one pass over the data) and rerun. Change one thing at a time, so you know
what made the difference. Set it back and rerun before Lab 5.

<p align="center">**[END OF LAB]**</p></br></br>

---

## **Lab 3 - LoRA: Train a Small Add-On Instead**

**Purpose: Train only about 2% of a model 4 times bigger than Lab 2's, and save
a small add-on file instead of a whole copy.**

**Time: 10 minutes**

<br><br>

**The idea.** LoRA leaves the model unchanged and trains a small add-on beside
it, called an *adapter*.

<br><br>

1. Merge every highlighted block:

```
code -d extra/lab3_lora_complete.txt finetune/lab3_lora.py
```

<br><br>

2. Run it (about 3½ minutes):

```
python finetune/lab3_lora.py
```

While it runs: **STEP 1** says about **2%** of the model will be trained
(Lab 2: 100%). The next line shows the model is now graded only on its answer,
not on repeating the ticket back. During training, watch its answer to test
ticket 1 change, as in Lab 2.

<br><br>

3. Read the line after training: the adapter is a few tens of MB; a full copy
would be about 1,400 MB. One shared model, plus a small adapter per customer
or task.

Then it opens **out/lab3_answers.txt** and fills it in as the trained model
answers: each test ticket, with the best prompt's, Lab 2's and Lab 3's answers.
Which tickets did Lab 3 fix? Which part does every model still get wrong?

![Lab 3 answers file](./images/ft-lab3-answers.png?raw=true "Each ticket with the best prompt's, Lab 2's and Lab 3's answers")

<br><br>

4. Read **RESULTS**:

![Lab 3 results](./images/ft-lab3-results.png?raw=true "Lab 3 next to Lab 2 and the best prompt")

Lab 3 beats Lab 2 and the best prompt on category. (The model changed too, so
this is the workshop's progress, not LoRA's alone.)

<br><br>

**Optional (outside the 10 minutes):** in your file, change `r=16` to `r=8`
and `lora_alpha=32` to `16` (a smaller adapter) and rerun. Set them back and
rerun before Lab 5, which uses this adapter.

<p align="center">**[END OF LAB]**</p></br></br>

---

## **Lab 4 - QLoRA: Train on a 4-Bit Model, Then Fold It In**

**Purpose: Store the model at half a byte per number to save memory, train an
adapter on top, then fold everything into one model to ship.**

**Time: 10 minutes**

<br><br>

**Before you run it:** on a CPU, 4-bit training is slow, so this lab trains
only 3 steps: enough to see it work, not to score it. **QLoRA saves memory,
not time.**

<br><br>

1. Merge every highlighted block:

```
code -d extra/lab4_qlora_complete.txt finetune/lab4_qlora.py
```

<br><br>

2. Run it (about 3 minutes):

```
python finetune/lab4_qlora.py
```

<br><br>

3. Read **STEP 1**: the same model in memory at 32 bits and at 4 bits, and how
much a model 3 times bigger would need at 4 bits. Less than Lab 3's model at
32 bits.

Then it opens **out/lab4_numbers.txt**. Read it while STEP 2 trains: a few of
the model's numbers before and after storing them in 4 bits. Each comes back
rounded, and small ones often become exactly 0.

![Lab 4 numbers file](./images/ft-lab4-numbers.png?raw=true "A few of the model's numbers before and after 4 bits")

<br><br>

4. Read **STEP 2**: the loss still falls, so training works on the 4-bit
model. Then the time per example, compared with Lab 3. Use QLoRA when memory
is the problem.

<br><br>

5. Read **STEP 3**: the adapter folded into one ordinary model. Fold it in for
one dedicated use; keep adapters separate to share one model across many
customers.

![Lab 4 output](./images/ft-lab4-results.png?raw=true "Memory at 32 and 4 bits, training time, and the folded-in model")

<br><br>

**Optional, at home:** a fair quality test: set `QLORA_STEPS = 24` and
`QLORA_BATCH = 8` (about 70 minutes on a CPU, minutes on a GPU).

<p align="center">**[END OF LAB]**</p></br></br>

---

## **Lab 5 - Compare Everything, and Check for Damage**

**Purpose: Put every run side by side, see where the best model fails, and
check that training didn't make it worse at everything else.**

**Time: 10 minutes**

<br><br>

1. Merge every highlighted block:

```
code -d extra/lab5_eval_complete.txt finetune/lab5_eval.py
```

<br><br>

2. Run it (about 2 minutes):

```
python finetune/lab5_eval.py
```

<br><br>

3. Read **STEP 1**, every run side by side. Compare each trained run with the
**best prompt** row, and read the MB column: what you would ship.

![Lab 5 comparison](./images/ft-lab5-results.png?raw=true "Every run side by side")

<br><br>

4. While **STEP 2** has the Lab 3 model answer the test tickets, it opens
**out/lab5_mistakes.txt**: every ticket the model gets at least partly wrong,
next to the correct answer. Read it while this step and the quiz run. Which
part is wrong most often?

![Lab 5 mistakes file](./images/ft-lab5-mistakes.png?raw=true "Every ticket the Lab 3 model got at least partly wrong")

Then read **STEP 2**: right answers for each category, and the most common
mix-up. An overall score says *if* there is a problem; this says *where*. The
usual fix: more training examples of the weak category.

<br><br>

5. Read **STEP 3**: a 20-question general-science quiz, before and after
training. The block you merged quizzes the same model twice: first with the
adapter switched off (the original model), then with it on. Small changes are luck. A **big drop** would mean training damaged
the model's other skills.

<br><br>

6. **Think about it:** which model would you put into production for
Northwind, and which two numbers from today support it?

<br><br>

**Optional (outside the 10 minutes): compare with a big hosted model.**
**STEP 4** sends the same tickets to a very large model hosted by Groq, with the
best prompt. It needs a free key: follow "Optional: a free Groq API key" in
`README.md`, then rerun. If a hosted model wins, you may not need to fine-tune.
Check that first.

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
| **End-of-text / padding token** | The end-of-text token is the stop signal the model must learn. The padding token is filler that makes a group of examples the same length. They must be *different* tokens (a comment in Lab 2's code explains why). |
| **Masking / -100** | Marking positions with -100 so they aren't graded — used so the model is graded only on its answer, not on repeating the ticket. |
| **Base vs. instruct model** | A base model only continues text. An instruct model had a further training pass to follow instructions. distilgpt2 is base; SmolLM2-360M-Instruct is instruct. |
| **Rank (r)** | The size of a LoRA adapter. Bigger can learn more, but is bigger and slower to train. |
| **Alpha (lora_alpha)** | How strongly the adapter's adjustments count. The usual setting is twice r. |
| **Adapter** | The small add-on LoRA trains (tens of MB), saved separately from the unchanged model. |
| **Quantization / nf4** | Storing a model's numbers in fewer bits to save memory. "nf4" is the 4-bit format QLoRA uses. |
| **Merge** | Folding an adapter into the model itself to get one ordinary model. Always merge into a full-size (16- or 32-bit) copy. |
| **Test set** | Tickets the model never trained on. The only scores that count. |
| **Baseline** | The best score you can get without training — what every later run is measured against. Ours is the Lab 1 best-prompt score. |
| **Catastrophic forgetting** | Training hard on a narrow task degrading everything else the model could do. Lab 5's general-knowledge quiz (STEP 3) is the check. |

<br><br>

---

<p align="center">
<b>For educational use only by the attendees of our workshops.</b>
</p>
<p align="center">
<b>(c) 2026 Tech Skills Transformations and Brent C. Laster. All rights reserved.</b>
</p>
