---
spec_version: "1.0"
course_title: "Fine-Tuning Small AI Models: LoRA, QLoRA & Evaluation"
course_version: "1.0"
created: "2026-07-31"
updated: "2026-07-31"
status: complete
interview_rounds_complete: [1, 2, 3, 4]
---

# Course Spec: Fine-Tuning Small AI Models: LoRA, QLoRA & Evaluation

**Pitch:** A 3-hour, CPU-only, hands-on walk through the complete fine-tuning lifecycle — dataset
prep, baseline eval, full fine-tune, LoRA, QLoRA, adapter merge, and systematic evaluation — using
models small enough to train in minutes on a free 4-vCPU GitHub Codespace.

**Type:** new

## Audience

- Who: Developers, data engineers, ML-curious platform engineers, technical leads evaluating
  fine-tuning for enterprise use cases.
- Prerequisites: Basic Python proficiency; conceptual understanding of what an LLM is. No prior
  ML training experience. No GPU.
- Class size / context: Virtual, half-day (3 hours), instructor-led with hands-on labs.

## Schedule

- Duration: half-day (3 hours / 180 minutes)
- Lab budget: 5 labs, 10 minutes max each (reading, merging, running and reading the results), 4-6 steps each, about 150 printed words each (2026-11-09)
- Sections: 5 lecture sections mapped 1:1 to the 5 labs, plus a wrap-up
- Suggested break points: one 10-minute break after Lab 2 (at the 1:13 mark)

### Agenda

| Time | Block | Minutes |
|------|-------|---------|
| 0:00 | Welcome, agenda, Codespace launch & environment check | 10 |
| 0:10 | **Section 1** — Why fine-tune? Fine-tune vs. prompt vs. RAG; the decision framework | 20 |
| 0:30 | **Lab 1** — Curate & format a Hugging Face dataset; run the baseline | 12 |
| 0:42 | **Section 2** — Dataset formats, chat templates, Trainer & TRL; what full fine-tuning costs | 19 |
| 1:01 | **Lab 2** — Full fine-tune distilgpt2 (82M) | 12 |
| 1:13 | **BREAK** | 10 |
| 1:23 | **Section 3** — LoRA: the low-rank idea, rank, alpha, target modules, DoRA | 20 |
| 1:43 | **Lab 3** — LoRA fine-tune SmolLM2-360M with PEFT + HF `Trainer` | 12 |
| 1:55 | **Section 4** — Quantization & QLoRA: NF4, double quant, paged optimizers, merge & export | 20 |
| 2:15 | **Lab 4** — QLoRA on SmolLM2-360M + merge/export | 12 |
| 2:27 | **Section 5** — Systematic evaluation: task metrics, harnesses, regression & forgetting | 17 |
| 2:44 | **Lab 5** — compare every run + before/after general quiz (lm-evaluation-harness, inside the script) | 12 |
| 2:56 | Wrap-up, where to go next, Q&A | 4 |

Total: welcome 10 + lecture 96 + labs 60 + break 10 + wrap-up 4 = 180 min.

## Topic Map

| # | Topic | Depth | Priority | Source | Notes |
|---|-------|-------|----------|--------|-------|
| 1 | Fine-tune vs. prompt-engineer vs. RAG decision framework | mastery | must-cover | abstract | Slides + Lab 1 framing |
| 2 | What fine-tuning actually changes (weights vs. context) | hands-on | must-cover | abstract | |
| 3 | Dataset curation: instruction/completion pairs | mastery | must-cover | abstract | Lab 1 |
| 4 | Chat templates & conversational formats | hands-on | must-cover | abstract | Lab 1 |
| 5 | Hugging Face `datasets` (load, split, map, inspect) | hands-on | must-cover | abstract | Lab 1 |
| 6 | Baseline evaluation before training | mastery | must-cover | abstract | Lab 1, Lab 5 |
| 7 | HF `Trainer` / `TrainingArguments` | hands-on | must-cover | abstract | Lab 2 |
| 8 | Full fine-tuning mechanics and cost | hands-on | must-cover | abstract | Lab 2 (distilgpt2 is 82M — the 6-layer distillation of GPT-2's 124M) |
| 9 | TRL `SFTTrainer` / `SFTConfig` | awareness | must-cover | abstract | Slides + Lab 3 callout only — see Assumptions Log |
| 10 | LoRA theory: low-rank decomposition, r, alpha, dropout | mastery | must-cover | abstract | Lab 3 |
| 11 | LoRA target modules (arch-specific) | hands-on | must-cover | abstract | Lab 3 |
| 12 | DoRA, rsLoRA (awareness + one live toggle) | awareness | must-cover | abstract | Lab 3 optional step |
| 13 | Quantization: NF4, blockwise, double quant | mastery | must-cover | abstract | Lab 4 |
| 14 | QLoRA via bitsandbytes on CPU | hands-on | must-cover | abstract | Lab 4 |
| 15 | Adapter merging & export (`merge_and_unload`) | hands-on | must-cover | abstract | Lab 4 |
| 16 | EleutherAI lm-evaluation-harness | hands-on | must-cover | abstract | Lab 5 |
| 17 | Task-specific metrics & schema compliance | mastery | must-cover | abstract | Labs 1 & 5 |
| 18 | Catastrophic forgetting / capability regression | hands-on | must-cover | research-gap (Critical) | Lab 5 — the point of running a general harness |
| 19 | Serving adapters (vLLM/TGI multi-adapter) | awareness | nice-to-have | research-gap (Important) | Slides only |
| 20 | Preference tuning (DPO/GRPO) as the next step | awareness | nice-to-have | research-gap (Important) | Wrap-up slide only |
| 21 | Fine-tune vs. hosted frontier API: the cost/latency/privacy trade | mastery | must-cover | user request | Lab 5 optional Step 6 (Groq) |

Rejected gap proposals: RLHF/PPO deep dive (out of scope for 3 hours); multi-GPU/FSDP (no GPU);
GGUF/llama.cpp export (mentioned on a slide, not lab'd).

Obsolete topics flagged: `DataCollatorForCompletionOnlyLM` (removed from TRL 1.x — masking is now
automatic); `load_in_4bit=True` as a `from_pretrained` kwarg (removed in transformers v5 — must use
`BitsAndBytesConfig`); `tokenizer=` on Trainer (now `processing_class=`).

## Lab Plan

| Lab | Status | Topic(s) | Style | Files (skeleton → complete) | Multi-process? | Interactive LLM? | Est. time |
|-----|--------|----------|-------|------------------------------|----------------|------------------|-----------|
| 1 | new | Dataset curation, HF datasets, chat template, baseline eval, task metrics | diff-merge + run-and-observe | `finetune/lab1_prep.py` → `extra/lab1_prep_complete.txt` | no | yes (short CPU generation) | 12 min |
| 2 | new | Full fine-tuning with HF Trainer on distilgpt2 | diff-merge + run-and-observe | `finetune/lab2_fullft.py` → `extra/lab2_fullft_complete.txt` | no | yes | 12 min |
| 3 | new | LoRA with PEFT + HF Trainer on SmolLM2-360M; explicit prompt masking; rank/alpha/targets; DoRA toggle | diff-merge + run-and-observe | `finetune/lab3_lora.py` → `extra/lab3_lora_complete.txt` | no | yes | 12 min |
| 4 | new | QLoRA (NF4 4-bit + LoRA), memory comparison, 4-step training probe + CPU cost measurement, merge & export | diff-merge + run-and-observe | `finetune/lab4_qlora.py` → `extra/lab4_qlora_complete.txt` | no | yes | 12 min |
| 5 | new | lm-evaluation-harness, custom task metrics, before/after table, optional frontier baseline | diff-merge + run-and-observe | `finetune/lab5_eval.py` → `extra/lab5_eval_complete.txt` | no | yes | 12 min |

- Capstone: Lab 5 combines every prior artifact (baseline, full-FT model, LoRA adapter, QLoRA
  merged model) into one comparison table plus a general-capability regression check.
- Reuse sources: none — new repo.

### Running task (used by all 5 labs)

**Enterprise support-ticket triage.** Input: a raw support ticket. Required output: a strict
one-line schema —

```
category: <one of 8> | priority: <P1-P4> | queue: <one of 6>
```

Chosen because it is (a) exactly the "proprietary classification schema" case from the abstract,
(b) something an 82M model cannot do at all before training and does well after, and (c) scorable
with crisp, non-subjective metrics: schema-validity rate, category accuracy, priority accuracy.
The dataset ships in-repo as JSONL so no live-workshop download can fail, and is loaded through
`datasets.load_dataset("json", ...)` so students still learn the real Hugging Face workflow.

## Slide Plan

| Section | Status | Slide count (est.) | Precedes lab(s) | Visuals needed |
|---------|--------|--------------------|-----------------|----------------|
| Title / About / Agenda / Setup | new | 6 | — | Title, about-me, agenda timeline, Codespace flow |
| 1. Why fine-tune? | new | 9 | Lab 1 | Decision-tree diagram, prompt/RAG/FT comparison cards, cost ladder |
| 2. Data & full fine-tuning | new | 9 | Lab 2 | Dataset-format cards, chat-template anatomy, Trainer flow, memory math bar |
| 3. LoRA | new | 10 | Lab 3 | Low-rank decomposition diagram, W + BA math, rank/alpha dial cards, target-module map, DoRA |
| 4. QLoRA | new | 8 | Lab 4 | Bit-width ladder, NF4 blockwise diagram, memory stack comparison, merge/export flow |
| 5. Evaluation | new | 8 | Lab 5 | Eval pyramid, harness flow, before/after table, forgetting curve |
| Wrap-up | new | 4 | — | Recap fits-together card stack, what's next, thanks |

Delivered deck: `Fine-Tuning-Small-AI-Models-LoRA-QLoRA-Evaluation.pptx`, 50 slides,
speaker notes on every slide, built on the `tst-template.pptx` master.

- Theme source deck: `tst-template.pptx` (supplied by the instructor, not stored in this repo; TechUpSkills, "2_Custom Design" theme, single slide
  master, Poppins titles `#002060`, teal accent `#00A79D`, navy `#1B3A6B`, card fill `#EDF2F7`)
- Visual density: diagram-heavy
- Animated build-ups: none (virtual delivery; static composed diagrams instead)
- Speaker script: notes on every slide

## Environment

- Delivery: GitHub Codespaces (free tier) — devcontainer also works locally
- hostRequirements: 4 CPU / 16gb / 32gb (standard free Codespace)
- Ollama model(s): none — this course trains models directly, no Ollama
- Key dependencies (pinned, minimum versions):
  `torch>=2.9` (CPU wheel index), `transformers>=5.14`, `peft>=0.20`, `trl>=1.9`,
  `datasets>=5.0`, `accelerate>=1.14`, `bitsandbytes>=0.50`, `lm-eval>=0.4.12`,
  `evaluate>=0.4.6`, `scikit-learn>=1.5`
- Models, all pre-fetched at Codespace create time by `scripts/prefetch_models.sh`:
  `distilgpt2` (82M — the 6-layer distillation of GPT-2's 124M; Labs 1-2),
  `HuggingFaceTB/SmolLM2-360M-Instruct` (362M; Labs 3-5),
  `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B; Lab 4 Step 8, memory measurement only).
  `microsoft/Phi-3-mini-4k-instruct` (3.8B) appears as a clearly-marked **optional take-home**
  step in Lab 4 — a 7.6 GB download and multi-minute CPU steps put it outside a 12-minute
  live lab, and it needs a `target_modules` change because it fuses its projections
  (`qkv_proj`, `gate_up_proj`). This substitution for the abstract's stated ladder is recorded
  in the Assumptions Log below.
- Services/ports: none
- Tokens (both optional, neither required): `HF_TOKEN` raises Hub rate limits and unlocks gated
  models; `GROQ_API_KEY` enables Lab 5's optional Step 6 frontier comparison against
  `openai/gpt-oss-120b`. Both are set as Codespaces secrets and injected as env vars. Every
  lab runs to completion with neither set.

## Conventions

Standard TechUpSkills conventions apply (repo structure, labs.md format, skeleton/diff-merge rules,
devcontainer template, slide master rules, copyright). Deviations for this course:

- No Ollama / no `startup_ollama.sh` — this course trains models locally rather than serving them.
- `postCreateCommand` additionally runs `scripts/prefetch_models.sh` so no lab step waits on a
  model download during the live session.

## Assumptions Log

- Lab domain (support-ticket triage): assumed — chosen to match the abstract's "proprietary
  classification schemas" while staying CPU- and metric-friendly. Unconfirmed.
- Model ladder changed from the abstract's distilgpt2 / TinyLlama-1.1B / Phi-3-mini to
  distilgpt2 82M / SmolLM2-360M / TinyLlama-1.1B, with Phi-3-mini as an optional take-home.
  Measured on a 4-core CPU: a full LoRA pass over TinyLlama-1.1B needs ~11 min and Phi-3-mini
  is a 7.6 GB download, both outside a 12-minute lab. SmolLM2-360M keeps the "progressively
  larger" progression (82M → 362M → 1.1B) while every lab still finishes in budget. Unconfirmed.
- Labs use the plain HF `Trainer` rather than TRL's `SFTTrainer`. Measured on a real
  4-core Codespace: `SFTTrainer` 226 s/step vs `Trainer` 7.3 s/step for the identical
  model, shapes and LoRA config (24 steps = 90 min vs 3 min). Prompt masking is therefore
  done explicitly in Labs 3-4. TRL remains in requirements.txt and on the slides as the
  GPU-path tool. Confirmed by QA run 2026-08-01.
- Lab 4 trains for **4 steps at batch 2**, not Lab 3's 24 steps at batch 8, and is framed as a
  mechanics-and-memory lab rather than a quality comparison. Measured on a real 4-core
  Codespace: bitsandbytes 4-bit NF4 runs 176.8 s/step at batch 8 and 43.0 s/step at batch 2,
  against 7-10 s/step for float32 — roughly 20x per training example, because the
  dequantize-and-multiply path is a CUDA kernel with no CPU equivalent. A like-for-like 24-step
  4-bit run measures ~71 minutes. The controlled quality experiment is therefore set as
  homework, and the QLoRA paper's near-zero-gap result is cited rather than demonstrated.
  Confirmed by QA run 2026-08-01.
- Lab 5's optional frontier comparison moved from `llama-3.3-70b-versatile` to
  `openai/gpt-oss-120b`. Groq **decommissioned** the Llama model on 2026-08-16; `gpt-oss-120b`
  is Groq's own recommended replacement and is Production (not Preview) tier. Because gpt-oss
  models always reason and reasoning tokens come out of the output budget, the call now sends
  `reasoning_effort="low"`, `reasoning_format="hidden"` and `max_completion_tokens=256` instead
  of `max_tokens=48`. The model ID lives in one constant (`FRONTIER_MODEL` in
  `common/triage.py`) precisely because this will happen again. Confirmed 2026-10-05 against
  console.groq.com/docs/deprecations.
- Abstract states distilgpt2 is 124M; the model is actually 81.9M parameters (124M is GPT-2
  small, which DistilGPT2 was distilled from). Materials use 82M throughout. Unconfirmed.
- Dataset shipped in-repo rather than pulled from the Hub: assumed for live-session reliability.
  Unconfirmed.
- Deck theme inherited from `tst-template.pptx`: confirmed 2026-07-31 (user supplied the file).

## Open Items

- Lab screenshots: `images/` placeholders are referenced in `labs.md` and need to be captured
  against a real Codespace run (use the `lab-screenshot-capturer` skill).
- Optional: publish the ticket dataset to the Hub so Lab 1 can also demo `load_dataset("<user>/...")`.
