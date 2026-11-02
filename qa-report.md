# Lab Test Report — Fine-Tuning Small AI Models

**Repository:** https://github.com/skillrepos/fine-tuning
**Runs:** v1.0 and v1.1 on 2026-08-01 (commit `c196332`); v1.2 built 2026-08-01 (not run);
**v1.3 content review 2026-10-05** (desk review + live library introspection, not run)
**Environment:** GitHub Codespaces, 4 vCPU (2 physical cores + SMT) / 15 GB, AMD EPYC 9V74

---

# v1.3 CONTENT REVIEW — 2026-10-05

A full accuracy / currency / flow review, two months after v1.2 was built. **Desk review
against primary sources plus live introspection of the current library stack; no Codespace
run.** Everything below is fixed in the files unless marked otherwise.

## Blocking — the workshop was broken as shipped

**1. The Groq model in Lab 5 Step 6 no longer exists.** Groq **decommissioned
`llama-3.3-70b-versatile` on 2026-08-16**. Every run of that step since has 404'd. Confirmed
against [console.groq.com/docs/deprecations](https://console.groq.com/docs/deprecations).

Fixed by moving to **`openai/gpt-oss-120b`** — Groq's own recommended replacement, and
Production rather than Preview tier so it should not vanish again mid-quarter. Three
consequences handled:

- The model ID now lives in **one constant**, `FRONTIER_MODEL` in `common/triage.py`, because
  this will happen again. Groq retired four model families in 2026 alone.
- `gpt-oss` models **always reason** and reasoning tokens come out of the *output* budget, so
  the old `max_tokens=48` would have returned **empty strings** with `finish_reason="length"`.
  The call now sends `reasoning_effort="low"`, `reasoning_format="hidden"` and
  `max_completion_tokens=256`, and the script warns explicitly if any reply comes back empty.
- The SDK (groq 1.x) already retries 429s twice honouring `Retry-After`, so the hand-rolled
  backoff was replaced with `Groq(max_retries=5)`. groq 1.x is otherwise **API-compatible**
  with the 0.11 the code was written against — the major version bump broke nothing.

The failure is also now *used*: Step 6's "Control — the model can be deprecated underneath
you" bullet cites this exact decommission by name, against a Lab 3 adapter that still runs.

**2. `README-Codespace.md` told attendees to print their own API tokens.** The secret check was

```
echo "HF_TOKEN is ${HF_TOKEN:+set}${HF_TOKEN:-NOT set}"
```

`${VAR:-default}` expands to **the value** when the variable is set, so this prints the live
token into the terminal — and into scrollback, screen shares and recordings. Verified by
execution. Replaced with a `[ -n "$VAR" ]` test and a short warning explaining the trap.

## Accuracy

**3. Three deck slides still said Labs 3-5 use TRL.** The v1.1 TRL→`Trainer` rewrite never
reached the deck. Slide 4 and slide 33 were titled "LoRA with TRL", and slide 18 stated
outright *"This is what Labs 3-5 use"* — contradicting slide 20, slide 33's own bullet, and
every lab. `README.md`'s lab table and `course-spec.md`'s agenda had the same error. All
corrected; slide 30 (the legitimate TRL reference slide) retitled **"The GPU path: TRL +
SFTTrainer"** so its role is unambiguous.

**4. Phi-3 was listed as Llama-style for `target_modules` — it is not.** Phi-3 fuses Q/K/V
into `qkv_proj` and the MLP into `gate_up_proj`, so the Llama names match nothing on it. This
appeared in the Lab 3 module table (labs.md and the lab script), deck slides 29 and 49,
`anticipated-qa.md` and `course-spec.md` — and the take-home instruction "change `BASE_MODEL`
to Phi-3-mini, not one other line changes" was simply wrong. Corrected everywhere, with a new
fused-projection row in the table, and the take-home now names the one edit required.

**5. PEFT 0.21 changed the failure mode the slides teach.** A `target_modules` list that
matches nothing now raises `NoMatchingPeftModuleError` instead of silently adapting zero
layers. Slide 29's punchline — *"it fails quietly, not loudly"* — had become backwards. Updated
to the real behaviour, including the asymmetry that a **partial** match still fails quietly,
which is why the trainable-count check remains the safety net. New Q&A entry added.

**6. Stale figures.** Lab 3's masking example said `49 tokens, 32 masked` where the measured
run prints `53 / 36 / 17`. Codespace build time said 8-12 min in both READMEs and ~11 min in
the Q&A timing table; measured after the base-image fix is **~4 min**. The Lab 4 OOM advice
told students to drop `per_device_train_batch_size` to 4, a setting Lab 4 no longer has
(it is `QLORA_BATCH = 2`).

## Currency — the stack moved, the code did not need to

Verified by installing the current versions and introspecting the real code, not by reading
release notes alone.

| Package | Written against | Current | Verdict |
|---|---|---|---|
| transformers | 5.14.1 | **5.18.0** | All 10 APIs unchanged. `warmup_steps` as a float-ratio confirmed in `training_args.py`; `processing_class`, `use_cpu`, `eval_strategy`, `stop_strings`, `dtype=` all still correct |
| peft | 0.20.0 | **0.21.2** | Unchanged, except finding 5 above |
| bitsandbytes | 0.50.0 | **0.50.2** | CPU NF4 path untouched; 0.50.1/0.50.2 are GPU-only work. Benchmarked both: no systematic CPU change, so the documented 43 / 176.8 s/step figures stand |
| torch | 2.13 | **2.14.1** | Benchmarked: no CPU regression |
| datasets | 5.0.0 | **5.1.0** | Unchanged |
| lm-eval | 0.4.12 | **0.4.13** | `lm_eval run`, `peft=`, `arc_easy`/`arc_challenge` all still valid — CLI exercised directly |
| groq | 0.11.0 | **1.7.0** | API-compatible; the *model* was the problem, not the SDK |

**No lab code changes were required for any of it.** `requirements.txt` now carries **upper
bounds** so a future major release cannot rewrite the labs under a class, with a note recording
the one hard constraint: peft 0.21.x requires transformers ≥ 5.17, and peft 0.20 breaks
encoder-decoder models on transformers ≥ 5.18 — so 5.14.1+0.20.0 and 5.18.0+0.21.2 are the
safe pairs, not a mix. `groq` is pinned `>=1.0,<2`. Resolution verified with a pip dry-run.

## Tooling recovered

`make_skeletons.py` — which generates `finetune/*.py` from `extra/*_complete.txt` — lived
outside the repo and was lost when the previous session's container was reclaimed. It has been
**reconstructed by diffing each complete/skeleton pair**, verified by round-trip (all 4/4/4/3/4
gaps recovered exactly), and is now **committed into the repo** at `scripts/make_skeletons.py`
with its gap spec in `scripts/lab_gaps.json`. It takes a `--check` flag suitable for CI.
Hand-editing `finetune/*.py` is now the wrong move and the docstring says so.

## Verified clean

- **Dataset integrity**, programmatically: 281 train / 71 eval, every label parses against
  `SCHEMA_RE`, every label's fields agree with its own columns, all categories/priorities/queues
  are in the declared label space, and **category → queue is a strict function** matching the
  routing table in the zero-shot prompt exactly. Zero defects.
- All 5 completed files and all 5 skeletons compile; skeletons byte-match what the generator
  produces.
- Models used by the labs are unchanged upstream (distilgpt2, SmolLM2-360M-Instruct,
  TinyLlama-1.1B-Chat-v1.0) — **not re-verified live**, see below.
- Lab timings, lab ordering, break placement and the 180-minute budget are untouched and
  internally consistent.
- The deck's QLoRA numbers, the Lab 5 `steps` column and the forgetting-slide deltas — all the
  v1.2 corrections — survived intact.

## Still open

1. **Nothing has been run end-to-end.** This sandbox reaches PyPI but the Hugging Face Hub is
   blocked at the proxy (403), so no weights can be downloaded. **Lab 4 has still never
   executed in its v1.2 shape**, and Lab 5 Step 6 has never run against `gpt-oss-120b`. One
   Codespace pass would close both.
2. **The six screenshots are still placeholders** — `images/` holds `SCREENSHOT PENDING`.
3. **v1.2 and v1.3 are not on GitHub.** `skillrepos/fine-tuning` is still at `c196332`
   (v1.1), which contains the 71-minute Lab 4. The local repo's git is also broken — OneDrive
   dehydrated `.git/refs/heads/main`, so `git` reports "your current branch appears to be
   broken". The working-tree files are all intact and readable; only git's internals are
   affected. Per your instruction this review touched no git state.

---

# v1.2 CHANGES (built 2026-08-01) — AWAITING A LIVE RE-RUN

Everything the v1.1 run found is now fixed in the materials. **Labs 1, 2, 3 and 5 are
unchanged in substance** (Lab 5 changed only its narration and one table column), so the
v1.1 PASS results still stand for them. **Lab 4 was reshaped and has not yet been run
end-to-end** — it needs one Codespace pass once these changes are pushed.

## Lab 4 redesign — as built

I kept the proposal's Steps 1-4 and 8 and its 4-step / batch-2 training probe, but changed
the merge step from what was proposed:

| Step | v1.1 | v1.2 as built |
|---|---|---|
| 1-4 memory, `Linear4bit`, `prepare_model_for_kbit_training` + LoRA | keep | **unchanged** |
| 5 train | 24 steps @ bs 8 (**71 min — unusable**) | **4 steps @ bs 2 (~2.9 min)**, framed as a mechanics probe; prints its own s/step and reads Lab 3's out of `results.json` for a live, same-machine comparison |
| 6 merge & export | merge the QLoRA adapter | **merge the QLoRA adapter (proposal said Lab 3's — changed deliberately)** |
| 7 evaluate | quality comparison vs Lab 3 | **evaluate the merged 4-step model and say plainly the scores are a step-count result, not a quantization result** |
| 8 TinyLlama 4-bit memory | keep | **unchanged** |

**Why I did not merge Lab 3's adapter as proposed.** Lab 3's adapter was trained against a
*float32* base. Merging it into bf16 demonstrates a merge, but not *the* rule the lab exists
to teach — "train against 4-bit, merge into full precision." Substituting a fp32-trained
adapter would have quietly broken the one thing Step 6 is for. Merging the real (if
undertrained) QLoRA adapter keeps the lesson exact, and the poor Step 7 scores are turned
into their own teaching point: **step count dominates every knob discussed all afternoon.**

The consequence, stated openly in the materials rather than papered over: **the workshop no
longer demonstrates the quantization-quality claim live.** It cites the QLoRA paper's
near-zero gap and sets the controlled experiment as homework (`QLORA_STEPS = 24`,
`QLORA_BATCH = 8`, ~70 min overnight). That is the honest outcome of the measurement.

## Corrections applied everywhere

| Was | Now | Files |
|---|---|---|
| "4-bit is ~1.4x slower on CPU" | measured **43 s/step @ bs 2, 177 s/step @ bs 8** vs **~8-10 s/step** fp32 — **~20x per training example**; QLoRA framed as a GPU technique that buys memory, not speed | `labs.md`, `extra/lab4_qlora_complete.txt`, `anticipated-qa.md`, deck (QLoRA-stack slide + Lab 4 slide + speaker notes) |
| Lab 5 Step 5 narration assumed **negative** harness deltas | rewritten for the measured reality (`arc_challenge +0.000`, `arc_easy +0.050`): near-zero is expected, a small positive is **noise at `--limit 20`**, a large negative is the finding you ran it to catch | `extra/lab5_eval_complete.txt`, `labs.md`, deck forgetting slide |
| Lab 5 table had no `steps` column, so row 4 read as a quality result | **`steps` column added**; row relabelled `Lab 4 QLoRA probe` | `extra/lab5_eval_complete.txt`, `labs.md`, deck results-table slide |
| Adapter size quoted as "33 MB" | **37 MB** (measured 36.5) | `extra/lab5_eval_complete.txt`, `anticipated-qa.md`, deck (×6) |
| `nproc` doc fix (from v1.1) | already applied locally, now in this drop | `README.md`, `README-Codespace.md` |

Deck rebuilt: 50 slides, 313 KB. All 5 skeletons regenerated from the completed files and
compile clean (gap counts 4 / 4 / 4 / 3 / 4). Assumptions Log updated in `course-spec.md`.

## What still needs doing

1. **Run Lab 4 end-to-end in a Codespace** — it has never executed in its new shape. I could
   not verify it here: this sandbox reaches PyPI but the Hugging Face Hub is blocked (`403`
   at the proxy), so no model weights can be downloaded. Static checks only: all files
   compile, and the Step 5 timing arithmetic and Lab 5 table were simulated against
   realistic values.
2. **Re-run Lab 5** afterwards, so its table picks up the new `qlora_smollm2` record.
3. **Capture the six screenshots** — `images/` still holds `SCREENSHOT PENDING` placeholders.

---

# v1.1 RE-RUN RESULTS (superseded for Lab 4)

**Codespace build: ~4 min** (was ~11 — the `python:3.11-bookworm` base image fix worked).

| Lab | Exit | Wall clock | Verdict |
|---|---|---|---|
| 1 — dataset prep & baseline | 0 | **30 s** | PASS |
| 2 — full fine-tune distilgpt2 | 0 | **173 s** | PASS |
| 3 — LoRA (rewritten onto `Trainer`) | 0 | **238 s** | **PASS — fix confirmed** |
| 4 — QLoRA | 143 (killed) | 272 s | **FAIL — new blocking defect** |
| harness: base | 0 | 65 s | PASS |
| harness: +adapter (`peft=`) | 0 | 69 s | PASS |
| 5 — evaluation incl. live Groq | 0 | **63 s** | PASS |

## The TRL fix is confirmed on real weights

Lab 3 ran at **7.34 s/step** (`10/24 [01:13<01:42, 7.34s/it]`) against 226 s/step under
TRL — a 31x improvement, exactly matching the isolated benchmark. Every documented figure
reproduced:

- `trainable params: 8,683,520 || all params: 370,504,640 || trainable%: 2.3437` (docs say ~2.3%)
- masking: `53 tokens, 36 masked as prompt, 17 graded`
- adapter **36.5 MB** vs full copy **1380.2 MB** → **38x smaller** (docs say ~40x)
- `pad id = 0, eos id = 2` — the PAD/EOS fix holds on SmolLM2

## Measured task metrics (real, n=20)

| Run | schema | category | priority | queue | MB | sec |
|---|---|---|---|---|---|---|
| Lab 1 baseline distilgpt2 82M | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| Lab 2 full FT distilgpt2 82M | 100.0% | 65.0% | 45.0% | 70.0% | 316 | 148 |
| Lab 3 LoRA fp32 SmolLM2 360M | 100.0% | 85.0% | 45.0% | 85.0% | 36.5 | 170 |
| Lab 4 QLoRA nf4 | (not run) | | | | | |

The intended narrative holds: LoRA on the 4x larger model beats full fine-tuning on category
(85% vs 65%) and queue (85% vs 70%) with a **38x smaller artifact**. Lab 5 printed
`(not run)` for the missing row — the graceful-degradation path works.

## Groq step works live

20 predictions from `llama-3.3-70b-versatile` in **5.4 s**. Head-to-head printed correctly:

| metric | LoRA 360M | llama-3.3-70b | delta |
|---|---|---|---|
| schema_valid | 100.0% | 100.0% | +0.0% |
| category_acc | 85.0% | 100.0% | +15.0% |
| priority_acc | 45.0% | 60.0% | +15.0% |
| queue_acc | 85.0% | 100.0% | +15.0% |

The frontier model wins — which is the outcome the lab text is explicitly written to handle
("if the frontier model wins, the honest read is this task may be prompt-able"). No change needed.

---

# NEW BLOCKING DEFECT: 4-bit training on CPU is ~24x slower than fp32

Lab 4 is **already on `Trainer`**, so this is a second, independent problem from the TRL one.

| Config | s/step | 24 steps |
|---|---|---|
| fp32, bs=8 (Lab 3) | **7.3** | ~3 min |
| 4-bit NF4, bs=8 (Lab 4) | **176.8** | **~71 min** |
| 4-bit NF4, bs=2 | **43.0** | ~17 min |

Per token, bitsandbytes 4-bit is roughly **24x slower** than fp32 on this CPU — not the
~1.4x the earlier desk research reported. That figure came from a micro-benchmark on a
4-layer toy model; on a real 360M model with 224 `Linear4bit` modules, dequantize overhead
dominates completely. **My materials state the 1.4x figure in labs.md, the deck and the
Q&A — all wrong and needing correction.**

## Proposed Lab 4 redesign (needs sign-off — it changes the lab's shape)

Keep everything that is fast and is the actual QLoRA lesson; stop pretending we can train
to convergence in 4-bit on a CPU.

| Step | Change |
|---|---|
| 1-4 memory, `Linear4bit` inspection, `prepare_model_for_kbit_training` + LoRA | **keep as-is** — fast, and this is the real lesson |
| 5 train | **4 steps at batch 2 (~2.9 min measured)**, framed explicitly as "proving gradients flow through quantized weights", not convergence |
| 6 merge & export | merge **Lab 3's** properly-trained adapter into a bf16 base — still teaches the correct-merge rule, with a real adapter |
| 7 evaluate | evaluates the merged Lab 3 adapter, so the Lab 5 table gets a real row |
| 8 TinyLlama 4-bit memory | **keep** |
| Lab 5 table | QLoRA row becomes a **memory + throughput** comparison rather than a quality one |

Plus: correct the 1.4x claim to the measured ~24x everywhere, and make the honest point
that **QLoRA is a GPU technique** — on CPU it buys memory at a severe speed cost.

---

# OTHER FINDINGS FROM THE v1.1 RE-RUN

**Regression I introduced (already fixed locally, not yet pushed):** `OMP_NUM_THREADS=2`
makes GNU `nproc` report **2**, because nproc honours that variable. README-Codespace told
students to check `nproc` reports 4, so everyone would think they got the wrong machine.
Verified: `nproc=2`, `nproc --all=4`, `env -u OMP_NUM_THREADS nproc` = 4. Docs now use
`nproc --all` and explain the pin.

**Minor:** Lab 5 Step 5's narration assumes negative harness deltas ("small negative deltas
are normal"). Measured deltas were `arc_challenge +0.000`, `arc_easy +0.050` — positive,
because `--limit 20` is pure noise. The text should cover the no-measurable-change case.

---

# ORIGINAL v1.0 RUN (for history)

## Summary

| | Count |
|---|---|
| Labs passed | 2 (Labs 1, 2) |
| Labs failed | 1 (Lab 3 — blocking) |
| Labs not reached | 2 (Labs 4, 5 — blocked by the Lab 3 defect) |
| Blocking defects | 1 |
| Medium defects | 3 |
| Low defects | 2 |

## Overall Assessment

Labs 1 and 2 run correctly and produce exactly the documented output, including both
intentional teaching moments (the PAD/EOS collision and the format-vs-semantics gap).
**Lab 3 is a blocking failure:** TRL's `SFTTrainer` runs at 226 seconds per step on this
hardware, so its 24-step run would take ~90 minutes against a claimed ~2 minutes. Lab 4
uses the same TRL code path and is expected to fail identically; Lab 5 depends on Lab 3's
adapter and could not be reached.

The cause is isolated and a fix is verified working (see Error 1). The course is **not
ready for students** until Labs 3–5 are reworked and re-tested.

## Environment verification

| Check | Result |
|---|---|
| Machine type | 4-core / 15 GB — `hostRequirements` honoured, defaulted correctly |
| Physical cores | **2** (4 vCPU = 2 cores × 2 SMT threads) |
| `torch.get_num_threads()` | 2 (PyTorch ignores `OMP_NUM_THREADS=4`) |
| Stack versions | torch 2.13.0+**cpu**, transformers 5.14.1, peft 0.20.0, trl 1.9.2, datasets 5.0.1, bitsandbytes 0.50.0, lm_eval 0.4.12, groq 1.6.0 — all match `requirements.txt` |
| CPU-only torch wheel | Confirmed (`+cpu`) — `pysetup.sh` index pin works, no CUDA libs pulled |
| Model prefetch | 3.1 GB cached, all 3 models present |
| `HF_TOKEN` / `GROQ_API_KEY` | Both present |
| `.venv` auto-activation | Works — prompt shows `(.venv)` |

## Lab-by-Lab Results

### Lab 1 — Preparing a Dataset and Measuring a Baseline

**Status:** PASS · **Steps:** 10/10 · **Wall clock:** 25 s (doc claims 60–90 s)

| Step | Description | Status | Notes |
|------|-------------|--------|-------|
| 1 | `head`/`wc` the raw data | PASS | 281 train / 71 eval — matches doc exactly |
| 2 | `code finetune/lab1_prep.py` [VIEW] | PASS | File present, 4 `# TODO` markers as documented |
| 3 | `code -d` merge [MERGE] | PASS | Applied via `cp` |
| 4 | `python finetune/lab1_prep.py` [RUN] | PASS | Exit 0 in 25 s |
| 5 | Read STEP 1–2 output | PASS | DatasetDict + label histogram render correctly |
| 6 | Read STEP 3 prompt/completion | PASS | Format as documented |
| 7 | Read STEP 4 chat template | PASS | `<|im_start|>` markers rendered |
| 8 | Read STEP 5 lengths | PASS | min 42 / median 53 / p95 60 / max 64 — doc says "50–70", accurate |
| 9 | Read STEP 7–8 baseline | PASS | **All four scores 0.0%**, exactly as documented |
| 10 | Discussion | PASS | Informational |

`distilgpt2: 81,912,576 parameters (82M)` printed — confirms the 82M correction was right.

### Lab 2 — Full Fine-Tuning with the Hugging Face Trainer

**Status:** PASS · **Steps:** 10/10 · **Wall clock:** 163 s total, 139 s training (doc claims 2–4 min)

| Step | Description | Status | Notes |
|------|-------------|--------|-------|
| 1 | `code finetune/lab2_fullft.py` [VIEW] | PASS | 4 `# TODO` markers |
| 2 | `code -d` merge [MERGE] | PASS | Applied via `cp` |
| 3 | `python finetune/lab2_fullft.py` [RUN] | PASS | Exit 0, 163 s |
| 4 | STEP 1 param count | PASS | `81,913,344 trainable / 81,913,344 total (100.0000%)` |
| 5 | `<|pad|>` note | PASS | `pad id = 50257, eos id = 50256` — distinct, as designed |
| 6 | STEP 3 collator tails | **WARN** | See Defect 4 — the illustrative output doesn't match reality |
| 7 | Watch training loss | PASS | Converged to 1.4216 |
| 8 | STEP 5–6 time + size | PASS | 139 s, 316 MB artifact |
| 9 | STEP 7 evaluation | PASS | schema 100.0%, category 65.0%, priority 45.0%, queue 70.0% |
| 10 | Discussion | PASS | Informational |

The intended lesson landed precisely: schema validity jumps to 100% while category
accuracy lags at 65% — "learning a format is easy; learning the semantics needs more model."

### Lab 3 — LoRA Fine-Tuning with TRL

**Status:** FAIL (blocking) · **Steps:** 3/11 executed before abort

| Step | Description | Status | Notes |
|------|-------------|--------|-------|
| 1 | `code finetune/lab3_lora.py` [VIEW] | PASS | 4 `# TODO` markers |
| 2 | `code -d` merge [MERGE] | PASS | Applied via `cp` |
| 3 | `python finetune/lab3_lora.py` [RUN] | **FAIL** | Reached step 1/24 after 3m46s; 226 s/step; ETA 1h26m. Aborted. |
| 4–11 | — | NOT RUN | Blocked by step 3 |

Steps 1–5 of the script itself (model load, module listing, LoRA config, `get_peft_model`,
dataset prep) all executed correctly. `pad id = 0, eos id = 2` confirms the pad-token fix
works on SmolLM2. The failure is purely training throughput.

### Lab 4 — QLoRA · **NOT RUN**

Blocked. Uses the same `SFTTrainer` code path as Lab 3 and is expected to fail identically
(worse, since 4-bit adds ~1.4x on CPU).

### Lab 5 — Systematic Evaluation · **NOT RUN**

Blocked. Steps 2 and 4–8 require `out/smollm2-lora`, which Lab 3 never produced. The
optional Groq step (Step 6) was therefore also not exercised against the live API.

## Error Details

### Error 1 (BLOCKING): TRL `SFTTrainer` is ~31x slower than an equivalent `Trainer` loop

**Command:** `python finetune/lab3_lora.py`
**Expected:** 24 steps in ~2 minutes
**Actual:** 226 s/step, ETA 1 h 26 min for 24 steps

**Ruled out, with evidence:**

| Hypothesis | Test | Result |
|---|---|---|
| Memory pressure / swap | `free`, `vmstat` | Swap 0 B, 10 GB available — not it |
| Process hung | CPU-time delta over 30 s | 36 s CPU / 30 s wall = 1.2 cores busy — genuinely computing |
| Model or LoRA config too heavy | Raw PyTorch loop, identical model/shapes/LoRA | **7.3 s/step** — not it |
| `max_length=192` padding | Printed collated batch shape | `(8, 63)` — TRL pads to longest, not to max_length — not it |
| v5 default `adamw_torch_fused` | Forced `optim="adamw_torch"` | Still slow — not it |
| Thread oversubscription | `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2` | Still slow — not it |

**Verified fix:** replace `SFTTrainer` with HF `Trainer` plus explicit prompt masking
(`labels = [-100]*len(prompt_ids) + completion_ids`) and `DataCollatorForSeq2Seq`:

```
trainable params: 8,683,520 || all params: 370,504,640 || trainable%: 2.3437
HF Trainer + LoRA: 7.3s/step -> 24 steps = 2.9 min
```

Same model, same LoRA config, same data — **7.3 s/step vs 226 s/step**. It also confirms
the documented `~2.3%` trainable figure exactly.

**Note:** Lab 2 already uses HF `Trainer` and ran at 1.9 s/step, which is what made the
comparison obvious — the slow path is specific to TRL's `SFTTrainer` on CPU.

**Cost of the fix:** Labs 3–5 lose TRL's automatic prompt/completion masking, which was a
teaching point. The masking becomes explicit instead — arguably a better lesson, and the
TRL behaviour can stay as a slide/callout.

## Other Defects

**Defect 2 (MEDIUM) — `OMP_NUM_THREADS: "4"` is wrong.**
`.devcontainer/devcontainer.json:6`. The box has 2 physical cores. Measured: 4 threads is
**17% slower** than 2 (2.79 vs 2.39 s/step). PyTorch ignores the variable and picks 2
anyway, so today it is merely misleading — but it advertises the wrong intent and could
affect other libraries. Set to `"2"` or drop it.

**Defect 3 (MEDIUM) — timing claims are unverified for Labs 3–5 and wrong for Lab 1.**
Lab 1 claims 60–90 s; actual 25 s. Labs 3/4/5 timings are entirely invalid given Error 1.
Every timing figure in `labs.md`, `course-spec.md`, `anticipated-qa.md` and the deck's
agenda needs regenerating from measured values.

**Defect 4 (MEDIUM) — Lab 2 Step 6's illustrative output cannot occur.**
`labs.md:228-231` shows:
```
  input_ids tail : [..., 50257, 50257, 50257]
  labels tail    : [...,  -100,  -100,  -100]
```
Actual output was `input_ids tail : [18, 930, 16834, 25, 5369, 12, 15097, 50256]` and an
identical `labels tail`. The script prints row 0 of a 2-row batch, and row 0 happens to be
the longer sequence, so it has **no padding at all** and no `-100`. Students are told to
"compare the tails" and will see two identical lines. Fix: print the shorter row, or pick
the row with the most padding.

**Defect 5 (LOW) — README/README-Codespace overstate Codespace build time.**
Both say 5–8 minutes. Actual was ~11 minutes, because the `python:1` devcontainer feature
compiles CPython from source on top of `base:bookworm`. Either update the figure or switch
the base image to `mcr.microsoft.com/devcontainers/python:3.11`, which ships Python
prebuilt and would genuinely cut the build.

**Defect 6 (LOW) — six `images/*.png` are placeholders.**
Known and already flagged, repeated here for completeness. Real screenshots still pending.

## Tester Limitations (not lab problems)

- Labs 4 and 5 were not executed at all; their status is **unknown**, not "passing".
- The optional Groq step was never exercised against the live API in the Codespace. It was
  verified earlier in a sandbox both with no key (clean skip) and with a stubbed client.
- I aborted Lab 3 rather than letting it run 90 minutes to completion, so I did not observe
  its final accuracy figures.
- Terminal output was read via screenshots and `grep` over redirected logs; long outputs
  were filtered, so I did not read every line of every run.

## Recommendations

1. **Rewrite Labs 3, 4 and 5 to use HF `Trainer`** instead of TRL `SFTTrainer`, with
   explicit prompt masking. Fix verified at 7.3 s/step.
2. **Re-run the full QA pass** afterwards — Labs 3–5 have never completed end to end.
3. Regenerate every timing figure from measurement, not estimate.
4. Fix `OMP_NUM_THREADS`, the Lab 2 Step 6 example output, and the build-time claim.
5. Consider `devcontainers/python:3.11` as the base image to cut build time.
6. Capture the six real screenshots during the re-run.

---

© 2026 Tech Skills Transformations and Brent C. Laster. All rights reserved.
