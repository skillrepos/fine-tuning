# Anticipated Q&A: Fine-Tuning Small AI Models: LoRA, QLoRA & Evaluation

**Generated**: 2026-07-31
**Course version**: 1.0

Instructor prep. 50 questions attendees are likely to ask, with short answers you
can give from the front of the room. Timing and break guidance is in the appendix.

---

## Section 1: Why fine-tune? (Lab 1)

**Q: How many examples do I actually need?**
A: Below about 100 you are usually better off with few-shot prompting. 200-500 is
enough to teach a rigid output format, which is what today's labs do with 281.
For real behavior change plan on 1,000-10,000 curated examples. Data quality
matters far more than count — 500 clean, consistent examples beat 5,000 noisy ones.

**Q: Can't I just put the schema in the prompt?**
A: On a frontier model, often yes — and if that works you should do it, because it
is free and instant. Two things break it: small models cannot reliably follow a
multi-part instruction, and on any model you re-send, re-tokenize, and re-pay for
that prompt on every single request. Lab 1 shows an 82M model scoring 0% on a
schema it has been shown; fine-tuning takes it near the top.

**Q: What's the difference between fine-tuning and RAG, really?**
A: RAG changes what the model *knows* at inference time by injecting retrieved
text. Fine-tuning changes what the model *does* by moving the weights. Facts →
RAG. Behavior, format, style, tone → fine-tuning. Most production systems use
both, plus a short prompt on top.

**Q: Can I fine-tune GPT-5 or Claude?**
A: Some hosted providers offer fine-tuning APIs; the details and availability
change constantly, so check current docs. What is different about today's approach
is that you own the weights and the artifact, you can run it air-gapped, and there
is no per-token training or inference bill. That matters for regulated data.

**Q: Will fine-tuning make the model smarter?**
A: No. It changes behavior, not underlying capability. A 360M model fine-tuned on
reasoning data is still a 360M model. If your task genuinely needs more reasoning
power, you need a bigger model — fine-tuning cannot manufacture capability that
was never there.

**Q: Why is your dataset synthetic? Isn't that cheating?**
A: It is a deliberate teaching trade-off, and `data/README.md` says so openly.
Synthetic templated data is more learnable than real ticket data, which is what
lets an 82M model show a clear before/after in under three minutes on 4 CPU cores.
Real corpora are messier — ambiguous labels, class imbalance, multi-issue tickets.
The workflow transfers directly; the absolute numbers do not.

**Q: Can I use my own data in these labs?**
A: Yes, and it is the best follow-up exercise. Append JSONL rows to
`data/tickets_train.jsonl` in the same schema. Keep the `label` string format
exactly as-is — `common/triage.py` parses it with a strict regex and any drift
silently scores as invalid.

**Q: How do I know if my task is a good fine-tuning candidate before I invest?**
A: Write ten examples by hand. If you can state the rule that maps input to output
in a paragraph, try a prompt first. If the rule is "it should look like these 200
examples" and you cannot articulate it crisply — that is exactly the fine-tuning
case.

---

## Section 2: Data & full fine-tuning (Lab 2)

**Q: Why did we add a `<|pad|>` token instead of just using EOS?**
A: The collator sets every PAD position in the labels to `-100` so padding does not
contribute to the loss. If PAD and EOS share a token id, that also erases every
genuine EOS from the labels, so the model never learns to stop and generates until
it hits the token limit. Two distinct tokens avoids the whole class of bug. Lab 2
prints both ids so you can see it.

**Q: Then why don't Labs 3 and 4 add a `<|pad|>` token too?**
A: Same problem, cheaper fix. SmolLM2 aliases PAD to EOS by default, but unlike
GPT-2 it already ships a spare special token (`<|endoftext|>`), so we just point
PAD at that. No new token means no `resize_token_embeddings` — which matters in
Lab 4, where you cannot cleanly resize the embedding table of a model that has
already been quantized.

**Q: Why doesn't Lab 2 mask the prompt like Lab 3 does?**
A: Deliberate contrast. Lab 2 is plain causal-LM fine-tuning — the standard
baseline, where the model is graded on every token. Lab 3 masks the prompt
explicitly. On an 82M model the difference is real: capacity spent re-predicting
the ticket text is capacity not spent on the answer.

**Q: Why don't the labs use TRL's SFTTrainer? Isn't that the standard tool?**
A: It is, on a GPU, and the slides cover it. But measured on a 4-core Codespace,
`SFTTrainer` ran at **226 s/step** against **7.3 s/step** for the plain HF
`Trainer` with the identical model, shapes and LoRA config — 24 steps would have
taken 90 minutes instead of 3. We could not reproduce the overhead from
`max_length` (TRL pads to longest, not to max_length), the v5 `adamw_torch_fused`
default, or thread oversubscription. Rather than ship a lab that cannot finish, we
use `Trainer` and mask the prompt by hand — which is a better lesson anyway,
because students see what `SFTTrainer` would have been doing for them.

**Q: What learning rate should I use?**
A: Full fine-tuning: 1e-5 to 5e-5 for a run of any real length; Lab 2 uses 3e-4 only
because it is 72 steps. LoRA: 1e-4 to 1e-3 — 20 to 100x what a normal full
fine-tune tolerates, because only the adapters move and B starts at zero. If your loss is flat, the LR is too low; if it
spikes or goes NaN, too high.

**Q: How many epochs?**
A: Two to four for most supervised fine-tuning. Beyond that you are usually
memorizing rather than learning, and the harness deltas in Lab 5 start getting
worse. If more epochs keep helping, that is a signal you need more data, not more
passes over the same data.

**Q: My loss went down but the output is still wrong. What happened?**
A: Loss going down only proves the optimizer worked. Common causes: format
mismatch between training and inference (check you use the same prompt builder for
both), EOS masked out of the labels, too few steps, or the model learned the format
but not the semantics — which is exactly what Lab 2 demonstrates on purpose.

**Q: Is transformers v5 stable enough to build on?**
A: Yes, and the migration is largely mechanical renames. The gotcha is that it
removed several things without a deprecation cycle, so essentially every tutorial
written before it is subtly wrong. The full list is on the "Hugging Face Trainer"
slide and in `.github/copilot-instructions.md`.

**Q: `warmup_steps=0.05` — is that five percent or five hundredths of a step?**
A: Five percent. In transformers v5, `warmup_steps` accepts a float below 1 and
interprets it as a ratio of total steps. `warmup_ratio` still exists in 5.14 but warns on use and is
scheduled for removal in v5.2. This one bites people because it does not error — it just silently does
something different from what the old code meant.

**Q: Why 192 tokens? My documents are much longer.**
A: Attention cost grows quadratically with sequence length, so this is the single
biggest lever on CPU training time. Measured on the real data: min 42, median 53,
p95 60, max 64 tokens. For longer documents expect training time to grow faster
than linearly, and consider sequence packing to reduce padding waste.

---

## Section 3: LoRA (Lab 3)

**Q: How do I pick the rank?**
A: Start at r=16 with alpha=32. Formatting and style work fine at r=8. Go to r=32
or 64 if you need more genuine behavior change. Past 64 you are approaching full
fine-tuning cost without its benefits. And be honest with yourself: better data
almost always beats a rank sweep.

**Q: What does alpha actually do?**
A: The LoRA update is scaled by alpha/r. The convention alpha = 2r keeps the
effective strength constant as you change rank, so you can sweep rank without
re-tuning your learning rate. If you set alpha and r independently you are tuning
two coupled knobs and will get confused.

**Q: Should I target attention only, or the MLP too?**
A: The original LoRA paper targeted attention only. Current practice usually
includes the MLP projections as well — more adapter capacity for a modest
parameter increase, which matters on short runs like ours. `target_modules="all-linear"`
is a safe default when you are on an unfamiliar architecture.

**Q: Why does B start at zero?**
A: So that BA = 0 at step 0 and the model is bit-identical to the base on the first
forward pass. No initialization shock, no risk of degrading the model before
training even starts. A is Gaussian-initialized so gradients can flow.

**Q: Is LoRA as good as full fine-tuning?**
A: For adaptation — format, style, schema, tone — the literature reports roughly
90-95% of full fine-tuning quality at a fraction of the cost, and our Lab 2 vs.
Lab 3 comparison is consistent with that. For absorbing genuinely new knowledge,
full fine-tuning still wins. Most enterprise work is adaptation.

**Q: Does LoRA slow down inference?**
A: Slightly, if you keep the adapter separate — you are doing an extra small matmul
per adapted layer. Merging (Lab 4) removes that entirely at the cost of losing
adapter swapping. For most workloads the unmerged overhead is a few percent and
well worth the flexibility.

**Q: Can I stack multiple adapters?**
A: Yes. PEFT supports loading several adapters and combining them with weights
(`add_weighted_adapter`). It works better for related tasks than for unrelated
ones. It is a real technique but not a free lunch — measure it, do not assume it.

**Q: When is DoRA worth it?**
A: At low rank (r=4 to 8), where plain LoRA runs short on capacity. It splits the
weight into magnitude and direction and applies the low-rank update only to the
direction. At r=16 on an easy task the difference will be small or lost in noise.
Cost is a bit more time per step and a few more parameters.

**Q: My trainable share (STEP 4's percentage) came out much lower than yours. Why?**
A: Almost certainly your `target_modules` matched fewer layers than you intended.
That is the classic LoRA failure and it fails quietly — training runs happily and
nothing improves. Print the module names (Lab 3 Step 2) and check the count STEP 4
prints.

**Q: Can I fine-tune on top of an already fine-tuned model?**
A: Yes. Either merge the first adapter and train a new one on the merged model, or
keep training the same adapter on new data. The second is simpler; the first is
cleaner if the two tasks are unrelated.

---

## Section 4: QLoRA (Lab 4)

**Q: Does 4-bit hurt quality?**
A: The QLoRA paper's central claim is that NF4 plus LoRA matches 16-bit LoRA
fine-tuning on standard benchmarks, and that is the answer to give. We do *not*
demonstrate it live — see the two questions further down about why. Lab 4 keeps
the model, data, rank and LoRA config identical to Lab 3 so that the only
variable *would* be base precision, but the step count had to be cut to 4 to fit
the lab, so its scores are not a quality result.

**Q: Why isn't the memory reduction 8x?**
A: Embeddings and the LM head stay unquantized, and every 64-weight block carries
its own scale factor. On a 360M model those exceptions are a large slice of the
total — SmolLM2's tied embedding table alone is 47M of 362M parameters — so you
measure roughly 2.5-3x against bfloat16 and 5-6x against float32. The bigger the model, the closer you get to the
theoretical ratio — which is why QLoRA's headline numbers come from 7B-70B models.

**Q: Why is QLoRA so much slower than LoRA in the lab? Isn't 4-bit supposed to be faster?**
A: On CPU it is dramatically slower, and Lab 4 makes them measure it rather than
take my word for it. We measured **43 s/step at batch 2** and **177 s/step at
batch 8** for 4-bit NF4, against **~8-10 s/step at batch 8** for float32 in Lab 3
— roughly **20x per training example**. bitsandbytes' dequantize-and-multiply
path is a CUDA kernel; the CPU path has no equivalent, so you pay the
dequantization in full on every matmul. That is why Lab 4 trains for only 4
steps and is framed as a mechanics probe.

On a GPU the picture is completely different: the QLoRA paper reports roughly a
third slower than 16-bit LoRA, in exchange for fitting a 65B model on one 48 GB
card. **QLoRA buys memory, not speed** — reach for it when memory is the binding
constraint, and use plain LoRA when the model already fits.

**Q: So we never actually see what quantization costs in accuracy?**
A: Correct, and say so plainly — do not fake it. The controlled experiment (same
model, same data, same rank, 24 steps, fp32 base vs 4-bit base) is a ~70-minute
CPU run. It is set as homework at the end of Lab 4: `QLORA_STEPS = 24`,
`QLORA_BATCH = 8`, run overnight, compare against Lab 3's row in
`out/results.json`. The published expectation is near-zero gap — the QLoRA paper
found 4-bit NF4 with double quantization matching 16-bit LoRA on their
benchmarks. Quantization trades memory for speed, not memory for accuracy.

**Q: Then why are Lab 4's scores so bad in the Lab 5 table?**
A: Because it trained for 4 steps — 8 examples of supervision. That row is *not*
a QLoRA quality result and the script says so in its own output. Turn it into the
lesson it actually is: step count dominates every hyperparameter discussed all
afternoon, and undertraining is the most common reason a real fine-tune
disappoints. The `steps` column in the Lab 5 table exists specifically so nobody
misreads that row.

**Q: Does bitsandbytes really work without a GPU?**
A: Yes, as of the 0.50 line. CPU is an officially supported backend with native
kernels for NF4 dequantization, and the full forward and backward pass works —
which is why Lab 4 runs at all. It needs AVX2, which every GitHub Codespaces host
has. Some older local machines and some ARM Macs do not.

**Q: What is "double quantization"?**
A: The per-block scale factors are themselves fp32 numbers — one per 64 weights,
about 0.5 bits of overhead per parameter. Double quantization quantizes those
scales too, recovering roughly 0.4 bits per parameter. Small, free, and on by
default in our config.

**Q: Why NF4 instead of int4?**
A: int4's 16 levels are evenly spaced, which wastes resolution on the tails where
almost no weights live. NF4 places its 16 levels so each receives an equal share of
a normal distribution — dense in the middle, sparse in the tails. Since trained
weights are approximately normally distributed, that is information-theoretically
optimal for the same 4 bits.

**Q: Why merge into bfloat16 and not into the 4-bit base?**
A: Merging into quantized weights bakes the quantization error permanently into
your model, and it is essentially impossible to detect afterwards. Correct order is
always: train quantized → merge into full precision → re-quantize the merged result
if you need to.

**Q: Should I merge or keep the adapter separate?**
A: Merge for a single dedicated deployment, or when exporting to GGUF, AWQ, or
GPTQ. Keep separate for multi-tenant serving where you want one base model in
memory and per-customer adapters swapped per request, and when you iterate on the
adapter often. There is no universally right answer.

**Q: Can I actually run Phi-3-mini in these labs?**
A: Not in a 12-minute live lab — it is a 7.6 GB download and each CPU step takes
minutes. Lab 4 Step 11 is set up as a take-home. Point `BASE_MODEL` at TinyLlama
and nothing else changes at all. Phi-3 needs exactly **one** more edit, and it is
a good one to make them find: Phi-3 fuses its projections, so `target_modules`
becomes `["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]`. Leave the Llama
names in and PEFT raises `NoMatchingPeftModuleError` — Lab 3 Step 2's lesson
arriving to collect.

**Q: Someone pasted the Llama `target_modules` list into the distilgpt2 lab and
got `NoMatchingPeftModuleError`. Is that a bug?**
A: No — that is PEFT 0.21 doing you a favour. Until 0.21 a `target_modules` list
that matched nothing silently adapted zero layers: training ran happily to
completion, the trainable count was tiny, and nothing improved. It now raises.
Worth calling out live, because the old silent version is what most blog posts
and most of their existing muscle memory assume. Note the asymmetry: a list that
matches only *some* of its names still succeeds quietly, so the trainable-count
check in Lab 3 Step 4 is still the real safety net.

**Q: What about GGUF and Ollama?**
A: Convert the merged model with `llama.cpp`'s `convert_hf_to_gguf.py`, quantize it
with `llama-quantize`, then write a Modelfile pointing at the GGUF. It is a clean
path from today's Lab 4 output to something you can `ollama run`.

---

## Section 5: Evaluation (Lab 5)

**Q: Why run a general benchmark at all? I only care about my task.**
A: Because your eval set contains only your task, by construction — so it cannot
detect that you made the model worse at everything else. That matters the moment
the same model handles a second use case, or a user asks it something slightly off
script. It is one command run twice; there is no excuse for skipping it.

**Q: How much regression is acceptable?**
A: A point or two on a benchmark is the ordinary cost of specialisation and usually
a fine trade. A large drop means catastrophic forgetting — lower the learning rate,
cut epochs, or mix 10-20% general instruction data into your training set and
retrain.

**Q: Why is `--limit 20` OK?**
A: It is not, for publishing. Twenty items per task is a smoke test: enough to
catch a collapse, nowhere near enough to resolve a two-point difference. Drop the
flag for a real number and expect it to take considerably longer on CPU. Knowing
which of your numbers are indicative and which are conclusive is part of the job.

**Q: Which benchmark tasks should I use for my domain?**
A: Pick two or three that touch the capabilities you care about preserving.
`arc_easy` and `piqa` are cheap general-reasoning proxies. If you serve code, add a
code task; if you serve multilingual traffic, add one. The point is a consistent
tripwire, not a leaderboard entry.

**Q: The hosted 120B model in Step 6 beat our fine-tuned model. So why did we bother?**
A: Sometimes that is the correct finding, and it is worth saying so out loud —
this task is deliberately learnable, so a large model with a good prompt does well
on it. What does not change: you pay per token on every request forever (that
prompt is ~250 tokens re-billed every call), you eat a network round trip per
ticket, every ticket leaves your network, and the provider can deprecate or
silently update the model underneath you. Your adapter is a 37 MB file you own and
can run air-gapped. The point of Step 6 is that this comparison costs one API call
and should happen *before* you invest in fine-tuning, not after.

**Q: Our fine-tuned 360M model beat the 120B model. Is that real?**
A: On a narrow task with a rigid output contract, yes, and it is the normal result.
The big model is reasoning about your schema from a prompt it has never seen; the
small one has it in the weights. This is the whole enterprise case for fine-tuning:
narrow task, high volume, small model you own. Do not over-generalise it — that
same 360M model will lose badly on anything requiring real reasoning.

**Q: Do I need a Groq account for the workshop?**
A: No. Lab 5 Step 6 is the only thing that uses it, it prints a note and skips
without a key, and nothing downstream depends on it. The free tier at
console.groq.com allows 30 requests/minute and 8,000 tokens/minute; the step
spends 20 requests and roughly 6,000 tokens, so it fits — but a second run inside
the same minute will 429. Tell them to wait a minute rather than retrying.

**Q: Step 6 returned a 404 / "model not found". What happened?**
A: Groq retired the model. This is not hypothetical — `llama-3.3-70b-versatile`,
the model this workshop originally called, was decommissioned on **2026-08-16**,
and the step broke until one constant was changed. The fix is one line:
`FRONTIER_MODEL` in `common/triage.py`, checked against
[console.groq.com/docs/deprecations](https://console.groq.com/docs/deprecations).

Use it. It is the single best live demonstration of the "Control" bullet in that
same step: a hosted model can be switched off underneath you on the vendor's
schedule, while the 37 MB adapter they trained in Lab 3 still runs exactly as it
did the day they built it.

**Q: Step 6 returned empty predictions and scored 0%.**
A: That is the reasoning-token trap, and the script warns about it explicitly.
`gpt-oss` models always reason and cannot be told not to — reasoning tokens come
out of the *output* budget, so too small a `max_completion_tokens` gets spent
thinking and returns an empty `content` with `finish_reason="length"`. Raise
`max_completion_tokens` in `FRONTIER_KWARGS` (`common/triage.py`). We ship 256
with `reasoning_effort="low"`, which is comfortable for a one-line answer.

**Q: What about LLM-as-judge?**
A: Useful when the output is open-ended and there is no exact-match answer — style,
tone, summarization. Two caveats: judges have systematic biases (position, length,
self-preference), and you need to validate the judge against human labels before
you trust it. For a strict schema like today's, deterministic parsing is better,
cheaper, and reproducible.

**Q: My scores are different from the ones on your slides.**
A: Expected. Different CPU counts, thread scheduling, and library versions all
introduce variation, and a 20-item eval sample is noisy. What should reproduce is
the *shape*: the bare prompt near zero, trained runs beating the best prompt on
category and queue, and Lab 4's 4-step test scoring low.

**Q: How do I evaluate a task that has no exact answer?**
A: Layer it. Deterministic checks for anything you can parse (length, required
sections, forbidden terms, valid JSON), reference-based metrics like ROUGE or
BERTScore where you have gold outputs, an LLM judge with a written rubric for the
subjective part, and human spot-checks. Never rely on the judge alone.

---

## General / cross-cutting

**Q: Everything scored 0% after I trained. What went wrong?**
A: Almost certainly they ran the script before merging, or stopped merging
partway. VS Code frequently splits one logical change into two adjacent regions
(the deleted `# TODO` comment and the inserted code), so tell them to work top to
bottom until no highlighting is left rather than counting blocks. Also check they
edited the *right* pane. Second most common cause is a format mismatch between
training and inference.

**Q: My Codespace is much slower than the lab says.**
A: Run `nproc`. If it says 2 instead of 4, you were given a smaller machine and
every training time roughly doubles. You can delete the Codespace and recreate it
with the 4-core option.

**Q: The process was `Killed` during Lab 4.**
A: Out of memory. Lab 4 briefly holds two models. Close other processes and rerun;
if it persists, drop `per_device_train_batch_size` to 4.

**Q: Can I run all of this locally instead of in a Codespace?**
A: Yes. Use the dev container with Docker Desktop, or `scripts/pysetup.sh` in a
plain virtualenv. Give Docker at least 4 CPUs and 16 GB. On Apple Silicon, MPS
works for LoRA but bitsandbytes 4-bit support on ARM Macs is less reliable — Lab 4's
quantized steps may not run.

**Q: How does this change on a GPU?**
A: Drop `use_cpu=True` and everything else stays the same. Expect 50-100x faster on
a single consumer GPU, and 4-bit becomes a speed win as well as a memory one. You
will also be able to raise batch size and sequence length substantially.

**Q: What's the difference between SFT, DPO, and RLHF?**
A: SFT — what we did today — trains on (input, correct output) pairs. DPO trains on
(input, preferred output, rejected output) triples to tune for what people prefer
rather than merely what is correct. RLHF/PPO does the same thing through a learned
reward model and reinforcement learning; it is more powerful and much harder to run.
Standard pipeline is SFT first, then DPO. Both are in TRL.

**Q: Do I need a Hugging Face token?**
A: Not for this workshop — every model and dataset is public and ungated. A free
token only raises anonymous rate limits, and some models (Llama, Gemma) require
accepting a license before download.

---

## Appendix: timing and break points

### Lab timing (4-core Codespace)

Measured on a real 4-core Codespace (AMD EPYC 9V74, 2 physical cores + SMT):

| Lab | Total | of which is compute | Notes |
|-----|-------|---------------------|-------|
| 1 | 10-12 min | **25 s measured** | Most of it is the baseline generation |
| 2 | 10-12 min | **163 s measured** (139 s training) | Tell them to start it and keep reading |
| 3 | 10-12 min | ~5 min (7.3 s/step x 24, plus load) | +3 min if they do the optional DoRA step |
| 4 | 10-12 min | ~6-7 min | 4 training steps @ ~43 s (4-bit is ~20x slower per example); Step 8 loads a 2nd model |
| 5 | 10-12 min | ~4 min | Two harness runs, then the script |

Labs 3 and 4 are the tight ones. If you are running behind, the DoRA step in
Lab 3 and Step 8 of Lab 4 are both cuttable without breaking anything downstream.

**Codespace build takes about 4 minutes** (measured after the base-image fix;
it was ~11 before). Still have attendees create it before the session starts —
a class all building at once is slower than one person building alone.

### Break points

- **Primary break: after Lab 2, at the 1:15 mark.** Ten minutes. This is the
  natural seam — full fine-tuning is done, LoRA has not started.
- **Emergency buffer:** the Lab 3 DoRA step and the Lab 4 TinyLlama memory step
  (Step 8) are both cuttable without breaking anything downstream.
- **If you are running long:** compress Section 4 by skipping the blockwise/NF4
  detail slide and going straight from the precision ladder to the QLoRA stack.
- **If you are running short:** the per-class breakdown discussion in Lab 5 and the
  "Where to go from here" slide both expand comfortably.

### Pre-flight checklist for the instructor

1. Have attendees create their Codespace **before** the session starts — the first
   build takes 5-8 minutes.
2. Confirm `nproc` reports 4 and the prompt shows `(.venv)`.
3. Confirm models are cached: the verification snippet in `README-Codespace.md`
   should complete in seconds, not minutes.
4. Have `out/results.json` from your own run handy. If a participant's Lab 3
   fails, dropping your `results.json` in lets them do Lab 5 Step 1; Steps 2-4
   need the adapter itself, and the script now says so and exits cleanly rather
   than crashing. Copying your whole `out/smollm2-lora/` directory over restores
   the full lab.

---

<p align="center">
<b>(c) 2026 Tech Skills Transformations and Brent C. Laster. All rights reserved.</b>
</p>
