# AI assistant instructions for this repository

This is a hands-on training repository. GitHub Copilot's autocomplete is
**intentionally disabled** in the dev container so that participants type and
merge the code themselves — that is where the learning happens.

If you are asking an AI assistant about a file in this repo, use the template
below. It produces an explanation instead of a solution.

## Explain-this-app template

> Explain the file `<path>` in this repository using the following structure.
> Do not rewrite the code and do not fill in any `# TODO` markers.
>
> 1. **What it does** — one paragraph, plain language, no jargon.
> 2. **High-level flow** — the numbered STEP sections in order, one line each.
> 3. **Key building blocks** — for each imported class or function that does
>    real work (`LoraConfig`, `SFTTrainer`, `BitsAndBytesConfig`,
>    `DataCollatorForLanguageModeling`, ...): what it is, what it is
>    responsible for here, and what would change if it were removed.
> 4. **Data flow** — trace one support ticket from the JSONL file through
>    formatting, tokenization, the loss, and back out as generated text.
> 5. **Safe experiments** — three parameters I could change to learn something,
>    what I should expect to happen to the loss or the score, and roughly how
>    much extra time each would cost on a 4-core CPU.
> 6. **Debug checklist** — the five most likely reasons this script would fail
>    or produce a 0% score, and how to confirm each one.

## Ground rules for assistants working in this repo

- **Never fill in a `# TODO` marker.** Those are the exercises. Explain what
  the student needs to work out; do not write it for them.
- Skeleton files in `finetune/` and their completed counterparts in `extra/`
  must stay byte-identical apart from the intentional gaps. If you change one,
  change the other.
- This repo targets **transformers v5, peft 0.20+, trl 1.9+**. Several v4-era
  idioms are removed and will not work:
  - `Trainer(tokenizer=...)` → `processing_class=`
  - `evaluation_strategy` → `eval_strategy`
  - `no_cuda` → `use_cpu`
  - `warmup_ratio` → `warmup_steps` (a float below 1 is read as a ratio)
  - `torch_dtype=` → `dtype=`
  - `SFTTrainer(max_seq_length=...)` → `SFTConfig(max_length=...)`
  - `DataCollatorForCompletionOnlyLM` → removed; TRL masks prompts automatically
  - `from_pretrained(load_in_4bit=True)` → `quantization_config=BitsAndBytesConfig(...)`
- Everything must run **CPU-only**. Do not suggest `.cuda()`, `device_map="auto"`,
  `fp16=True`, or flash-attention.
- Keep runtimes short. Any suggestion that pushes a lab past ~4 minutes of
  training on 4 cores is not appropriate here.
