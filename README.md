# Fine-Tuning Small AI Models: LoRA, QLoRA & Evaluation

**A 3-hour hands-on workshop — Revision 1.4, 10/29/26**
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
| 3 | LoRA with PEFT + the Hugging Face `Trainer` | SmolLM2-360M-Instruct | LoRA (~2.3% of params) |
| 4 | QLoRA (4-bit NF4), adapter merge and export | SmolLM2-360M + TinyLlama-1.1B | QLoRA |
| 5 | lm-evaluation-harness, task metrics, regression check, optional frontier comparison | all of the above | evaluation |

## Prerequisites

- Basic Python proficiency
- A conceptual understanding of what an LLM is
- **No** prior ML training experience
- **No** GPU

### Accounts you need

| Account | Needed? | Why |
|---|---|---|
| **GitHub** (free) | **Yes** | To run the workshop's Codespace. Sign up at <https://github.com/signup> if you do not have one. |
| **Groq** (free, no credit card) | Optional | Only for Lab 5's optional Step 6. See [below](#optional-a-free-groq-api-key-for-lab-5-step-6). |

That is all. **No Hugging Face account or token is needed** — the labs only
*download* public models and datasets from Hugging Face; nothing is uploaded and
no Hugging Face cloud service is used. No OpenAI key, no payment method.

<br>

---

## Setup — GitHub Codespace (recommended)

Do this **before** the workshop starts. It takes about 10 minutes, mostly waiting.

**1. Change your Codespace's idle timeout from 30 minutes to something longer.**

By default a Codespace stops after 30 minutes without activity, which can happen
during a long lecture section. When logged in to GitHub, go to
<https://github.com/settings/codespaces>, scroll to **Default idle timeout**, set
it higher (the maximum is 240 minutes), and click **Save**.

![Changing the Codespace idle timeout](./images/ft-setup-timeout.png?raw=true "Changing the Codespace idle timeout")

<br>

**2. Click the button below to start a new Codespace from this repository.**

Click here ➡️  [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/skillrepos/fine-tuning?quickstart=1)

<br>

**3. Click "Create new codespace".**

![Creating a new Codespace](./images/ft-setup-create.png?raw=true "Creating a new Codespace")

The repository requires a **4-core / 16 GB** machine and GitHub picks it for you.
(If you are asked to choose, choose 4-core. The labs are timed for it, and Lab 4
needs the memory.)

If VS Code asks whether you trust the authors of the files, click
**Trust Folder & Continue**.

![Trust the workspace](./images/ft-setup-trust.png?raw=true "Trust the workspace")

The Codespace then installs the Python libraries and downloads the three models
the labs use (about 4 GB), so no lab has to wait on a download later. This takes
about **5-8 minutes**. It is ready when the terminal shows:

```
>>> Model prefetch complete.
```

<br>

**4. Check that everything works.**

In the Codespace **TERMINAL**, run:

```
python scripts/verify_setup.py
```

It takes under a minute. The last line should say
`RESULT: ready for the workshop.` (Any `WARN` lines are optional.)

<br>

**5. Open `labs.md`.**

Open **`labs.md`** from the file list on the left. It should open formatted; if
you see raw text instead, press `Ctrl+Shift+V` (`Cmd+Shift+V` on Mac). This is the
document you follow during the workshop.

**You are ready for the labs!**

<br>

**If your Codespace stops and you need to reopen it:** go to
<https://github.com/codespaces>, find the Codespace in the list, click the **...**
next to it, and choose **Open in browser**. Your files, the installed libraries,
and the downloaded models are all still there.

More detail and fixes for common problems: [README-Codespace.md](README-Codespace.md).

<br>

---

## Optional: a free Groq API key for Lab 5 Step 6

Lab 5 ends with an **optional** step that compares your fine-tuned model against a
very large model hosted by Groq. It needs a free Groq API key. **Every other lab
runs without one**, and if you skip this, that step simply prints a note and
moves on.

**a.** In a browser, go to <https://console.groq.com> and create an account. (If
you get an email with a button to confirm, make sure the link opens in the same
browser where you were using Groq. If not, copy the link and paste it into the
right browser.)

**b.** In the top right of the Groq screen, click **API Keys**.

![Groq API Keys menu](./images/ft-groq-1.png?raw=true "Groq API Keys menu")

**c.** Click **Create API Key**.

![Create API Key button](./images/ft-groq-2.png?raw=true "Create API Key button")

**d.** Give it any name, verify you're human if asked, and click **Submit**.

![Naming the key](./images/ft-groq-3.png?raw=true "Naming the key")

**e.** **Copy the key** — Groq will not show it to you again.

![Copy the key](./images/ft-groq-4.png?raw=true "Copy the key")

**f.** Back in the Codespace **TERMINAL**, run the command below, paste your key
when asked, and press *Enter*:

```
source scripts/setup-key.sh
```

This sets the key for this terminal and for every terminal you open later. (No
GitHub settings or Codespace secrets are involved.)

**g.** Confirm the key works:

```
python scripts/check_groq.py
```

You want to see a line starting with `OK`. If it reports that the model is not
available, Groq has retired it; the script lists what your key *can* reach and
tells you the one line to change.

<br>

---

## Can't use a Codespace?

You can run the labs on your own computer instead. These guides walk you through
installing everything the Codespace would have set up for you, and explain what
looks different outside the Codespace (including alternatives to the `code -d`
merge steps):

- **Mac:** [Local_Setup_Mac.md](Local_Setup_Mac.md) — Apple Silicon Macs (M1 or
  newer) on macOS 14+. Intel Macs are not supported; please use the Codespace.
- **Windows:** [Local_Setup_Windows.md](Local_Setup_Windows.md) — Windows 10/11
  on Intel or AMD processors. Windows-on-ARM PCs are not supported; please use
  the Codespace.

Both guides also describe a Docker Desktop option that builds the same
environment as the Codespace.

<br>

---

## System requirements

| | Codespace | Your own machine (minimum / recommended) |
|---|---|---|
| CPU | 4-core (chosen for you) | 2 cores / 4 or more |
| Memory | 16 GB | 8 GB / 16 GB |
| Disk | 32 GB | 10 GB free / 20 GB free |
| GPU | none | none — the labs deliberately run on the CPU |

Labs are timed against a 4-core Codespace. On other machines times will differ.

## Repository layout

```
.devcontainer/          Dev container / Codespace configuration
.github/                Copilot instructions for students
common/triage.py        Shared helpers: data loading, generation, scoring
data/                   The support-ticket dataset (JSONL)
extra/                  Completed versions of every lab file (.txt)
finetune/               Lab skeleton files students edit
images/                 Screenshots referenced by the docs
scripts/                Setup, model download, environment check, Groq key
labs.md                 *** THE LAB DOCUMENT — start here ***
Local_Setup_Mac.md      Running the labs on your own Mac
Local_Setup_Windows.md  Running the labs on your own Windows PC
README-Codespace.md     Codespace details and troubleshooting
course-spec.md          Course design spec (agenda, topic map, lab plan)
anticipated-qa.md       Instructor prep: likely student questions
requirements.txt        Python dependencies
```

### Useful scripts

| Command | What it does |
|---|---|
| `python scripts/verify_setup.py` | Checks libraries, downloaded models, 4-bit training, the `code` command, and the optional Groq key |
| `python scripts/prefetch_models.py` | Downloads the models (the Codespace does this for you; safe to re-run) |
| `source scripts/setup-key.sh` | Sets the optional Groq key (Codespace, Mac, Linux, Git Bash) |
| `.\scripts\setup-key.ps1` | Sets the optional Groq key (Windows PowerShell) |
| `python scripts/check_groq.py` | Confirms the Groq key and the Lab 5 model work |

## Running the labs

Open `labs.md` and follow it top to bottom.

Artifacts land in `out/` (it is not tracked in git). To start over from scratch:

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
The download script is safe to re-run and skips anything already downloaded:
```
python scripts/prefetch_models.py
```
If it fails with **`429 Too Many Requests`**, Hugging Face is briefly limiting
downloads (this can happen when a whole class downloads at the same moment).
Wait a few minutes and run it again — no account or token is needed.

**Training feels much slower than the lab says.**
Check your machine size with `nproc --all` — it should report 4. (Plain `nproc`
reports 2 here, because GNU nproc honours the `OMP_NUM_THREADS=2` this repo sets
deliberately; see README-Codespace.md.) A genuinely 2-core Codespace will roughly
double every training time in `labs.md`.

**`Killed` during Lab 4.**
Out of memory. Close other running processes and re-run; Lab 4 briefly holds
two models (Step 6 reloads the base at bfloat16 to merge into, and Step 8 loads
TinyLlama). If it persists, drop `QLORA_BATCH` from 2 to 1 and skip Step 8.

**Lab 4 fails to load the model in 4-bit.**
Run `python scripts/verify_setup.py` — its 4-bit check tells you whether this
machine can run Lab 4. Codespaces always can. On your own machine, the 4-bit
library needs an Intel/AMD processor with AVX2 (Windows, Linux) or an Apple
Silicon Mac on macOS 14+. If the check fails, use a Codespace for Lab 4 or follow
along with the instructor.

**Everything scores 0% after training.**
You almost certainly ran the file before merging in the completed code, or you
stopped merging partway. Re-run the `code -d` step and work top to bottom until
no highlighting is left, save, and run again. The **right** pane is yours to
edit; the left is the completed reference.

## License

See [LICENSE](LICENSE). These materials are for educational use only by
attendees of Tech Skills Transformations workshops.

© 2026 Tech Skills Transformations LLC and Brent C. Laster. All rights reserved.

- Web: [techskillstransformations.com](https://techskillstransformations.com) ·
  [getskillsnow.com](https://getskillsnow.com)
- Contact: training@getskillsnow.com
