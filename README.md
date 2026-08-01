# Fine-Tuning Small AI Models: LoRA, QLoRA & Evaluation

**A 3-hour hands-on workshop — Revision 1.0, 07/31/26**
Tech Skills Transformations LLC · Brent C. Laster

Pre-trained models are powerful generalists, but most enterprise use cases demand
domain-specific behavior: internal terminology, company style guides, proprietary
classification schemas, or regulatory language. This workshop teaches the complete
fine-tuning workflow — dataset preparation, training, and systematic evaluation —
using models small enough to train in minutes on a **CPU-only GitHub Codespace**.

## What you will build

Five labs, 10-12 minutes each, all on the same running task: turning free-text
support tickets into a strict proprietary schema.

| Lab | Topic | Model | Method |
|-----|-------|-------|--------|
| 1 | Dataset curation, chat templates, baseline evaluation | distilgpt2 (82M) | none |
| 2 | Full fine-tuning with the Hugging Face `Trainer` | distilgpt2 (82M) | all parameters |
| 3 | LoRA with PEFT + TRL `SFTTrainer` | SmolLM2-360M-Instruct | LoRA (~2.3% of params) |
| 4 | QLoRA (4-bit NF4), adapter merge and export | SmolLM2-360M + TinyLlama-1.1B | QLoRA |
| 5 | lm-evaluation-harness, task metrics, regression check, optional frontier comparison | all of the above | evaluation |

## Prerequisites

- Basic Python proficiency
- A conceptual understanding of what an LLM is
- **No** prior ML training experience
- **No** GPU
- A free GitHub account (for Codespaces)

No Hugging Face token, OpenAI key, or paid account is required. Every model and
dataset used here is public and free.

Two secrets are *optionally* supported and never required — see
[README-Codespace.md](README-Codespace.md#optional-codespaces-secrets):

- `HF_TOKEN` — raises Hub rate limits; useful when a whole class downloads at once
- `GROQ_API_KEY` — enables Lab 5's optional frontier-model comparison step

## Setup

### GitHub Codespaces (recommended)

1. Click **Code → Codespaces → Create codespace on main** in this repository.
2. Choose a **4-core / 16 GB** machine type.
3. Wait for the container to build. This takes **5-8 minutes** the first time —
   it installs the Python stack and pre-downloads all three models so no lab
   step waits on a download during the workshop.
4. You are ready when the terminal shows `>>> Model prefetch complete.`

See [README-Codespace.md](README-Codespace.md) for step-by-step screenshots and
troubleshooting.

### Local dev container

With Docker Desktop and the VS Code **Dev Containers** extension installed,
clone the repo, open it in VS Code, and choose **Reopen in Container** when
prompted. Allocate at least 4 CPUs and 16 GB of memory to Docker.

### Manual (no container)

```
python3 -m venv .venv
source .venv/bin/activate
bash scripts/pysetup.sh .venv
bash scripts/prefetch_models.sh
```

## System requirements

| | Minimum | Recommended |
|---|---|---|
| CPU | 2 cores | 4 cores |
| Memory | 8 GB | 16 GB |
| Disk | 20 GB | 32 GB |
| GPU | none | none |

Labs are timed against a 4-core machine. On 2 cores expect roughly double the
training times quoted in `labs.md`.

## Repository layout

```
.devcontainer/         Dev container / Codespace configuration
.github/               Copilot instructions for students
common/triage.py       Shared helpers: data loading, generation, scoring
data/                  The support-ticket dataset (JSONL)
extra/                 Completed versions of every lab file (.txt)
finetune/              Lab skeleton files students edit
images/                Screenshots referenced by labs.md
scripts/               Environment setup and model prefetch
labs.md                *** THE LAB DOCUMENT — start here ***
course-spec.md         Course design spec (agenda, topic map, lab plan)
anticipated-qa.md      Instructor prep: likely student questions
requirements.txt       Python dependencies
```

## Running the labs

Open `labs.md` from the file explorer and follow it top to bottom. Press
`Ctrl+Shift+V` (`Cmd+Shift+V` on Mac) to render it as a preview rather than raw
Markdown.

Artifacts land in `out/` (gitignored). To start over from scratch:

```
rm -rf out/
```

## Troubleshooting

**`ModuleNotFoundError` / wrong Python.**
Your terminal prompt should start with `(.venv)`. If not:
```
source .venv/bin/activate
```

**A model download hangs or fails.**
The prefetch script is safe to re-run and skips anything already cached:
```
bash scripts/prefetch_models.sh
```

**Training feels much slower than the lab says.**
Check your machine size — `nproc` should report 4. A 2-core Codespace will
roughly double every training time in `labs.md`.

**`Killed` during Lab 4.**
Out of memory. Close other running processes and re-run; Lab 4 briefly holds
two models. If it persists, reduce `per_device_train_batch_size` to 4.

**`bitsandbytes` import error or "no AVX2".**
The 4-bit CPU backend needs AVX2. All GitHub Codespaces hosts have it; some
older local machines and some ARM Macs do not. There is no partial workaround —
Steps 2 through 8 of Lab 4 all depend on the quantized load. Run that lab in a
Codespace, or follow along with the instructor and read the output together.

**Everything scores 0% after training.**
You almost certainly ran the file before merging in the completed code, or you
stopped merging partway. Re-run the `code -d` step and work top to bottom until
no highlighting is left, save, and run again. The **right** pane is yours to
edit; the left is the completed reference.

**Broken image icons in `labs.md`.**
Expected in this revision — lab screenshots are being captured against a live
Codespace run and are not in `images/` yet. Nothing else is affected.

## License

See [LICENSE](LICENSE). These materials are for educational use only by
attendees of Tech Skills Transformations workshops.

© 2026 Tech Skills Transformations LLC and Brent C. Laster. All rights reserved.

- Web: [techskillstransformations.com](https://techskillstransformations.com) ·
  [getskillsnow.com](https://getskillsnow.com)
- Contact: training@getskillsnow.com
