# Running the Labs on Your Own Windows PC (Without a Codespace)

**The recommended way to take this workshop is the GitHub Codespace described in
[README.md](README.md).** Use this guide only if you cannot use a Codespace.
It installs everything the Codespace would have set up for you, on your own PC.

Plan on **30-45 minutes** for the whole setup, most of it waiting for downloads.
Do it **before** the workshop — not during it.

> These steps have not been tested on every PC. Step 9 runs a check script that
> tells you, in under a minute, whether your machine can run every lab.

<br>

---

## Will this work on my PC?

Open **Settings → System → About** and look at **System type**:

| What you see | Can you use this guide? |
|---|---|
| **64-bit operating system, x64-based processor** (Windows 10 or 11) | **Yes.** |
| **64-bit operating system, ARM-based processor** (for example a Snapdragon laptop or Surface Pro X) | **No — please use the Codespace.** PyTorch does not publish a Windows-on-ARM version for the Python the labs use. |

You will also need about **10 GB of free disk space** (the three models are about
4 GB) and ideally **16 GB of memory** (8 GB works, but Lab 4 will be tight).
Lab 4 also needs a processor from roughly 2015 or later (one with "AVX2"); the
check in Step 9 tests this for you.

You do **not** need administrator rights for any step below except possibly
installing the programs in Steps 1-3.

<br>

---

## 1. Install Git (this also gives you "Git Bash")

1. Download Git for Windows from <https://git-scm.com/download/win> and run it.
2. Click **Next** through every screen — the default choices are fine.

Check it worked — open a **new** PowerShell window (press the Windows key, type
*PowerShell*, press Enter):

```
git --version
```

You should see something like `git version 2.51.0.windows.1`.

<br>

## 2. Install Python 3.11

The labs are tested on **Python 3.11** — the same version the Codespace uses.

1. Go to <https://www.python.org/downloads/windows/>, find **Python 3.11.9**
   (the last 3.11 release with an installer — newer Pythons are further up the
   page; skip them), and click **Download Windows installer (64-bit)**.
2. Run it. On the **first screen**, tick **"Add python.exe to PATH"** at the
   bottom. *This is the step people miss.*
3. Click **Install Now**. If the last screen offers **"Disable path length
   limit"**, click it — it prevents a rare install error later.

Check it worked — in a **new** PowerShell window:

```
py -3.11 --version
```

You should see `Python 3.11.something`.

> **Typing `python` opens the Microsoft Store?** That is a Windows shortcut,
> not real Python. Use `py -3.11` as shown here, or turn the shortcut off in
> **Settings → Apps → Advanced app settings → App execution aliases** (switch
> off both *python.exe* and *python3.exe*).

<br>

## 3. Install VS Code

The labs use VS Code's side-by-side diff view to merge code (`code -d ...`).

1. Download VS Code from <https://code.visualstudio.com/> and run the installer.
2. Keep **"Add to PATH"** ticked (it is by default). Ticking **"Add 'Open with
   Code' action"** is handy too.

Check it worked — in a **new** PowerShell window:

```
code --version
```

<br>

## 4. Allow PowerShell to run the workshop's scripts

Windows blocks PowerShell scripts by default, including the one that turns on
the Python environment. Run this once (no administrator rights needed) and type
`Y` if asked:

```
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

While you are here, tell Python to always use UTF-8 text, which avoids rare
"codec can't decode" errors on Windows:

```
[Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "User")
```

Close this PowerShell window and open a new one so both settings take effect.

<br>

## 5. Download the workshop files

Put the files somewhere that is **not** synced by OneDrive. On many PCs your
*Documents* and *Desktop* folders are synced by OneDrive automatically, and it
can fight with the large model files. Your home folder is a good choice:

```
cd $HOME
git clone https://github.com/skillrepos/fine-tuning.git
cd fine-tuning
```

From here on, **every command is run from inside this `fine-tuning` folder.**

<br>

## 6. Create the Python environment

This makes a private copy of Python just for this workshop, in a folder called
`.venv`, so nothing you install affects the rest of your PC.

```
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Your prompt should now start with **`(.venv)`**. That means the environment is
*active*.

> **Every time you open a new terminal for the workshop**, run `cd $HOME\fine-tuning`
> and then `.\.venv\Scripts\Activate.ps1` again. (The Codespace does this for you
> automatically. On your PC, you do it.)

<br>

## 7. Install the libraries

```
python -m pip install --upgrade pip
pip install "torch>=2.9"
pip install -r requirements.txt
```

(The first line must start with `python -m` on Windows — plain
`pip install --upgrade pip` fails because pip cannot replace itself while running.)

This downloads about 1 GB and takes roughly **3-8 minutes**. Lots of text scrolls
past — that is normal. It is finished when you get your `(.venv)` prompt back
without a red `ERROR:` line.

> The comment at the top of `requirements.txt` about a special "CPU-only"
> PyTorch download applies to Linux only. On Windows the plain
> `pip install torch` above is already the CPU version.

<br>

## 8. Download the models

```
python scripts\prefetch_models.py
```

This downloads the three models the labs use (about **4 GB**) plus two small test
datasets for Lab 5. It takes **5-15 minutes** depending on your connection.
It is finished when you see:

```
>>> Model prefetch complete.
```

No Hugging Face account or token is needed — every model is public. You may see a
warning about **"symlinks"** — it is harmless (it only means the download cache
uses a little more disk on Windows). If the download stops partway, just run the
same command again; it skips anything already downloaded.

<br>

## 9. Check that everything works

```
python scripts\verify_setup.py
```

This takes under a minute and needs no internet. It checks the libraries, the
downloaded models, the `code` command, and — most importantly — **actually runs a
tiny 4-bit training step**, which is the part of the workshop (Lab 4) most likely
to behave differently on your machine.

You want the last line to say:

```
RESULT: ready for the workshop.
```

Any `FAIL` line tells you what to fix. `WARN` lines are optional. A line
mentioning `gemm_4bit_forward` and `kernels` is a harmless message from the 4-bit
library; you will see it during the labs too. Ignore it.

<br>

## 10. (Optional) Set up a Groq key for Lab 5 Step 6

Lab 5 has one **optional** step that compares your model against a large hosted
model on Groq. It needs a free Groq API key. Every other lab runs without one.

To get a key, follow the screenshots in **[README.md, "Optional: a free Groq API
key"](README.md#optional-a-free-groq-api-key-for-lab-5-step-6)**. Then, in
PowerShell (with `(.venv)` active):

```
.\scripts\setup-key.ps1
```

Paste your key when asked and press Enter. Then confirm it works:

```
python scripts\check_groq.py
```

You want to see a line starting with `OK`. The script also saves the key as a
Windows user setting, so new terminals have it too — **but VS Code only notices
after you quit it completely and reopen it.** (To remove the key later: Start
menu → type *environment variables* → **Edit environment variables for your
account** → delete `GROQ_API_KEY`.)

<br>

## 11. Open the workshop in VS Code

```
code .
```

In VS Code:

1. If asked **"Do you trust the authors of the files in this folder?"**, click
   **Yes, I trust the authors**.
2. Open **`labs.md`** from the file list on the left and press `Ctrl+Shift+V` to
   see it formatted. This is the document you follow during the workshop.
3. Open a terminal inside VS Code with **Terminal → New Terminal** and activate
   the environment there too:

   ```
   .\.venv\Scripts\Activate.ps1
   ```

   (Optional: install the **Python** extension from Microsoft, press
   `Ctrl+Shift+P` → **Python: Select Interpreter**, and choose the one in `.venv`.
   VS Code will then activate it in new terminals for you.)

**You are ready for the workshop.**

<br>

---

## What is different from the Codespace

`labs.md` is written for the Codespace's Linux terminal. You have two choices on
Windows:

- **Option A — use Git Bash (easiest).** In VS Code's terminal panel, click the
  small **⌄** arrow next to the **+** and choose **Git Bash**. Almost every
  command in `labs.md` then works exactly as written. The one change: activate
  the environment with

  ```
  source .venv/Scripts/activate
  ```

  (Windows puts it in `Scripts`, not `bin`.) For the optional Groq key in Git
  Bash, use `source scripts/setup-key.sh`.

- **Option B — stay in PowerShell.** Most commands work as written too —
  `python ...`, `code ...` and `code -d ...` all do, forward slashes included.
  These are the ones that need changing:

| In `labs.md` (Codespace) | In PowerShell |
|---|---|
| `source .venv/bin/activate` | `.\.venv\Scripts\Activate.ps1` |
| `head -3 data/tickets_train.jsonl` | `Get-Content data\tickets_train.jsonl -TotalCount 3` |
| `wc -l data/tickets_train.jsonl data/tickets_eval.jsonl` | `(Get-Content data\tickets_train.jsonl).Count; (Get-Content data\tickets_eval.jsonl).Count` |
| `rm -rf out/` | `Remove-Item -Recurse -Force out` |
| `nproc` | `$env:NUMBER_OF_PROCESSORS` |
| Commands split over several lines, each ending in `\` | PowerShell uses a backtick `` ` `` at the end of a line instead of `\`. Or use the one-line versions below. |

The two multi-line commands in Lab 5 (Steps 1 and 2), as single lines you can
paste into PowerShell:

```
lm_eval run --model hf --model_args pretrained=HuggingFaceTB/SmolLM2-360M-Instruct,dtype=float32 --tasks arc_easy,arc_challenge --device cpu --limit 20 --batch_size 1 --output_path out/harness-base
```

```
lm_eval run --model hf --model_args pretrained=HuggingFaceTB/SmolLM2-360M-Instruct,peft=out/smollm2-lora,dtype=float32 --tasks arc_easy,arc_challenge --device cpu --limit 20 --batch_size 1 --output_path out/harness-lora
```

A few more differences, whichever terminal you use:

| In `labs.md` / the slides | On your PC |
|---|---|
| The terminal starts with `(.venv)` automatically | Activate it yourself in every new terminal (see above). |
| Times like "training takes 2-4 minutes" | Those are measured on a 4-core Codespace. Your PC may be faster or slower, and Lab 4's 4-bit training on Windows has not been timed by us. |
| Model files live in the repo's `.cache/` folder | They live in `C:\Users\<you>\.cache\huggingface`. |
| `OMP_NUM_THREADS` is set to 2 | Leave it unset on your PC — the default uses all your cores. |

### Merging the lab code (`code -d`)

Each lab has a step like:

```
code -d extra/lab1_prep_complete.txt finetune/lab1_prep.py
```

**If the `code` command works (Step 3), this is identical to the Codespace:** the
**left** side is the finished code, the **right** side is your file. Click the
small arrow next to each highlighted block to copy it from left to right, then
save with `Ctrl+S`.

To also get the short explanations that pop up when you hover over each block
(the Codespace installs this for you), install the workshop's small VS Code
extension once, from the `fine-tuning` folder:

```
code --install-extension .devcontainer/merge-info-0.1.1.vsix
```

If you prefer a different tool:

- **WinMerge** (free, <https://winmerge.org/>): open WinMerge, choose
  **File → Open**, pick the `extra\..._complete.txt` file on the left and your
  `finetune\...py` file on the right, then copy each difference to the right
  (**Merge → Copy to Right**) and save.
- **PyCharm:** select both files in the Project panel, then press `Ctrl+D`.
- **If you fall behind:** copy the whole finished file over yours. You skip
  reading the changes, but the lab runs:

  ```
  Copy-Item extra\lab1_prep_complete.txt finetune\lab1_prep.py
  ```

  (Change the two file names to match the lab you are on. In Git Bash use
  `cp extra/lab1_prep_complete.txt finetune/lab1_prep.py`.)

<br>

---

## Troubleshooting

**`running scripts is disabled on this system`**
Repeat Step 4, then open a new PowerShell window.

**`py` or `python` is not recognized**
Python is not installed, or "Add python.exe to PATH" was not ticked. Re-run the
Python installer, choose **Modify**, and make sure it is added to PATH. Then open
a new PowerShell window.

**`code` is not recognized**
Re-run the VS Code installer with **Add to PATH** ticked, then open a new
PowerShell window.

**`ModuleNotFoundError: No module named 'torch'` (or any other library)**
The environment is not active in this terminal. Run `.\.venv\Scripts\Activate.ps1`
— your prompt should start with `(.venv)`.

**`pip install` fails with an error mentioning a path that is too long**
Re-run the Python installer, choose **Modify → Next**, and tick **"Disable path
length limit"** (or ask your IT team to enable Windows long paths). Then delete
the `.venv` folder and repeat Steps 6 and 7.

**`pip install` fails with an `ERROR:` about a package version**
Make sure the environment was created with Python **3.11** (Step 6). Check with
`python --version` while `(.venv)` is active. If it shows another version, delete
the `.venv` folder and repeat Steps 6 and 7.

**Antivirus or company security software slows or blocks the install**
Some security tools scan every downloaded library file. If installs crawl or
fail, ask your IT team, or use the Codespace instead.

**`verify_setup.py` shows `FAIL` on the 4-bit test**
Lab 4 will not run on this PC. Labs 1, 2, 3 and 5 are unaffected. For Lab 4,
either use a Codespace or follow along on the instructor's screen.

**The PC slows to a crawl or a lab stops with a memory error**
Close other apps (browsers with many tabs especially) and run the lab again.

**Start a lab over from scratch**
Delete its output and re-run it: `Remove-Item -Recurse -Force out`

<br>

---

## Alternative: Docker Desktop

If you already use Docker Desktop (with WSL 2), VS Code can build the exact same
environment as the Codespace on your PC:

1. Install the **Dev Containers** extension in VS Code (`Ctrl+Shift+X`, search
   *Dev Containers*, by Microsoft).
2. In Docker Desktop → **Settings → Resources**, give Docker at least **4 CPUs**
   and at least **8 GB** of memory (12 GB or more is better for Lab 4).
3. Clone the repository (Step 5), run `code .`, and click **Reopen in Container**
   when VS Code offers it (or `Ctrl+Shift+P` → **Dev Containers: Reopen in
   Container**).
4. The first build takes 10-20 minutes. It is ready when the terminal shows
   `>>> Model prefetch complete.` Then everything in `labs.md` works exactly as
   written, and you can skip the "What is different" section above.

<br>

<p align="center">
<b>(c) 2026 Tech Skills Transformations and Brent C. Laster. All rights reserved.</b>
</p>
