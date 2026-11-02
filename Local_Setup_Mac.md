# Running the Labs on Your Own Mac (Without a Codespace)

**The recommended way to take this workshop is the GitHub Codespace described in
[README.md](README.md).** Use this guide only if you cannot use a Codespace.
It installs everything the Codespace would have set up for you, on your own Mac.

Plan on **30-45 minutes** for the whole setup, most of it waiting for downloads.
Do it **before** the workshop — not during it.

> These steps have not been tested on every Mac. Step 8 runs a check script that
> tells you, in under a minute, whether your machine can run every lab.

<br>

---

## Will this work on my Mac?

Click the **Apple menu  → About This Mac** and look at two lines:

| What you see | Can you use this guide? |
|---|---|
| **Chip: Apple M1 / M2 / M3 / M4** (or newer) **and macOS 14 Sonoma or newer** | **Yes.** |
| **Chip: Apple M…** but macOS 13 or older | Update macOS first. The 4-bit library Lab 4 needs requires macOS 14+. |
| **Processor: Intel …** | **No — please use the Codespace.** PyTorch and the 4-bit library no longer publish versions for Intel Macs. |

You will also need about **10 GB of free disk space** (the three models are about
4 GB) and ideally **16 GB of memory** (8 GB works, but Lab 4 will be tight).

<br>

---

## 1. Install Apple's command-line tools (this gives you Git)

Open the **Terminal** app (press `Cmd+Space`, type *Terminal*, press Enter) and run:

```
xcode-select --install
```

Click **Install** in the window that appears and wait for it to finish. If it says
the tools are *already installed*, that is fine — move on.

Check it worked:

```
git --version
```

You should see a version number such as `git version 2.39.5`.

<br>

## 2. Install Python 3.11

The labs are tested on **Python 3.11** — the same version the Codespace uses.

1. Go to <https://www.python.org/downloads/macos/>, find **Python 3.11.9**
   (the last 3.11 release with an installer — newer Pythons are further up the
   page; skip them), and click **Download macOS 64-bit universal2 installer**.
2. Open the downloaded `.pkg` file and click through the installer.
3. When it finishes, a Finder window opens on the **Python 3.11** folder.
   Double-click **Install Certificates.command** in that folder. (This lets
   Python download the models securely. Skipping it is the most common cause
   of `SSL: CERTIFICATE_VERIFY_FAILED` errors later.)

Check it worked — in a **new** Terminal window:

```
python3.11 --version
```

You should see `Python 3.11.something`.

> **Already use Homebrew?** `brew install python@3.11` works instead of the
> python.org installer, and you can skip the certificates step.

<br>

## 3. Install VS Code and its `code` command

The labs use VS Code's side-by-side diff view to merge code (`code -d ...`).

1. Download VS Code from <https://code.visualstudio.com/> and drag
   **Visual Studio Code** into your **Applications** folder.
2. Open VS Code.
3. Press `Cmd+Shift+P`, type **shell command**, and choose
   **Shell Command: Install 'code' command in PATH**.

Check it worked — in a **new** Terminal window:

```
code --version
```

<br>

## 4. Download the workshop files

Put the files somewhere that is **not** synced by iCloud, Dropbox or OneDrive —
those services can fight with the large model files. Your home folder is a good
choice:

```
cd ~
git clone https://github.com/skillrepos/fine-tuning.git
cd fine-tuning
```

From here on, **every command is run from inside this `fine-tuning` folder.**

<br>

## 5. Create the Python environment

This makes a private copy of Python just for this workshop, in a folder called
`.venv`, so nothing you install affects the rest of your Mac.

```
python3.11 -m venv .venv
source .venv/bin/activate
```

Your Terminal prompt should now start with **`(.venv)`**. That means the
environment is *active*.

> **Every time you open a new Terminal window for the workshop**, run
> `cd ~/fine-tuning` and then `source .venv/bin/activate` again. (The Codespace
> does this for you automatically. On your Mac, you do it.)

<br>

## 6. Install the libraries

```
pip install --upgrade pip
pip install "torch>=2.9"
pip install -r requirements.txt
```

This downloads about 1 GB and takes roughly **3-8 minutes**. Lots of text scrolls
past — that is normal. It is finished when you get your `(.venv)` prompt back
without a red `ERROR:` line.

> The comment at the top of `requirements.txt` about a special "CPU-only"
> PyTorch download applies to Linux only. On a Mac the plain `pip install torch`
> above is the right one.

<br>

## 7. Download the models

```
python scripts/prefetch_models.py
```

This downloads the three models the labs use (about **4 GB**) plus two small test
datasets for Lab 5. It takes **5-15 minutes** depending on your connection.
It is finished when you see:

```
>>> Model prefetch complete.
```

No Hugging Face account or token is needed — every model is public. If it stops
partway (a network hiccup), just run the same command again; it skips anything
already downloaded.

<br>

## 8. Check that everything works

```
python scripts/verify_setup.py
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

## 9. (Optional) Set up a Groq key for Lab 5 Step 6

Lab 5 has one **optional** step that compares your model against a large hosted
model on Groq. It needs a free Groq API key. Every other lab runs without one.

To get a key, follow the screenshots in **[README.md, "Optional: a free Groq API
key"](README.md#optional-a-free-groq-api-key-for-lab-5-step-6)**. Then, in your
Terminal (with `(.venv)` active):

```
source scripts/setup-key.sh
```

Paste your key when asked and press Enter. Then confirm it works:

```
python scripts/check_groq.py
```

You want to see a line starting with `OK`. The script saves the key in your
`~/.zshrc` file so new Terminal windows have it too. (To remove it later, delete
the `export GROQ_API_KEY=...` line from that file.)

<br>

## 10. Open the workshop in VS Code

```
code .
```

In VS Code:

1. If asked **"Do you trust the authors of the files in this folder?"**, click
   **Yes, I trust the authors**.
2. Open **`labs.md`** from the file list on the left and press `Cmd+Shift+V` to
   see it formatted. This is the document you follow during the workshop.
3. Open a terminal inside VS Code with **Terminal → New Terminal** and activate
   the environment there too:

   ```
   source .venv/bin/activate
   ```

   (Optional: install the **Python** extension from Microsoft, press
   `Cmd+Shift+P` → **Python: Select Interpreter**, and choose the one in `.venv`.
   VS Code will then activate it in new terminals for you.)

**You are ready for the workshop.**

<br>

---

## What is different from the Codespace

`labs.md` is written for the Codespace. On your Mac almost everything works as
written; these are the exceptions.

| In `labs.md` / the slides | On your Mac |
|---|---|
| The terminal starts with `(.venv)` automatically | Run `source .venv/bin/activate` in every new terminal (Step 5). |
| `Ctrl+S` to save, `Ctrl+Shift+V` to preview | `Cmd+S` and `Cmd+Shift+V`. |
| `nproc` to count CPU cores | `sysctl -n hw.ncpu` |
| Times like "training takes 2-4 minutes" | Those are measured on a 4-core Codespace. An Apple Silicon Mac is often faster for Labs 1-3. Lab 4's 4-bit training on a Mac has not been timed by us — it may be faster or slower. |
| Model files live in the repo's `.cache/` folder | They live in `~/.cache/huggingface`. |
| `OMP_NUM_THREADS` is set to 2 | Leave it unset on your Mac — the default uses all your cores. |
| Everything runs on the CPU | Same on your Mac. The labs deliberately use the CPU (`use_cpu=True`), not the Mac's GPU, so your numbers are comparable to everyone else's. Please don't change that during the workshop. |

### Merging the lab code (`code -d`)

Each lab has a step like:

```
code -d extra/lab1_prep_complete.txt finetune/lab1_prep.py
```

**If the `code` command works (Step 3), this is identical to the Codespace:** the
**left** side is the finished code, the **right** side is your file. Click the
small arrow next to each highlighted block to copy it from left to right, then
save with `Cmd+S`.

To also get the short explanations that pop up when you hover over each block
(the Codespace installs this for you), install the workshop's small VS Code
extension once, from the `fine-tuning` folder:

```
code --install-extension .devcontainer/merge-info-0.1.1.vsix
```

If you prefer a different tool:

- **PyCharm:** select both files in the Project panel, then press `Cmd+D`.
- **FileMerge** (comes with the full Xcode app from the App Store):
  `opendiff extra/lab1_prep_complete.txt finetune/lab1_prep.py -merge finetune/lab1_prep.py`
- **Any other diff tool:** compare the `extra/..._complete.txt` file (the finished
  version) with the `finetune/...py` file (yours), and copy the differences into
  yours.
- **If you fall behind:** copy the whole finished file over yours. You skip
  reading the changes, but the lab runs:

  ```
  cp extra/lab1_prep_complete.txt finetune/lab1_prep.py
  ```

  (Change the two file names to match the lab you are on.)

<br>

---

## Troubleshooting

**`command not found: python3.11`**
Python 3.11 is not installed, or you are in a Terminal window that was open
before you installed it. Open a new window and try again; if it still fails,
repeat Step 2.

**`command not found: code`**
Repeat Step 3 part 3, then open a **new** Terminal window.

**`SSL: CERTIFICATE_VERIFY_FAILED` while downloading models**
Run **Install Certificates.command** from your **Applications → Python 3.11**
folder (Step 2), then run `python scripts/prefetch_models.py` again.

**`ModuleNotFoundError: No module named 'torch'` (or any other library)**
The environment is not active in this terminal. Run
`source .venv/bin/activate` — your prompt should start with `(.venv)`.

**`pip install` fails with an `ERROR:` about a package version**
Make sure the environment was created with Python **3.11** (Step 5). Check with
`python --version` while `(.venv)` is active. If it shows another version, delete
the environment (`rm -rf .venv`) and repeat Steps 5 and 6.

**`verify_setup.py` shows `FAIL` on the 4-bit test**
Lab 4 will not run on this Mac. Labs 1, 2, 3 and 5 are unaffected. For Lab 4,
either use a Codespace or follow along on the instructor's screen.

**The Mac slows to a crawl or a lab prints `Killed`**
Out of memory. Close other apps (browsers with many tabs especially) and run the
lab again.

**Start a lab over from scratch**
Delete its output and re-run it: `rm -rf out/`

<br>

---

## Alternative: Docker Desktop

If you already use Docker Desktop, VS Code can build the exact same environment
as the Codespace on your Mac:

1. Install the **Dev Containers** extension in VS Code (`Cmd+Shift+X`, search
   *Dev Containers*, by Microsoft).
2. In Docker Desktop → **Settings → Resources**, give Docker at least **4 CPUs**
   and at least **8 GB** of memory (12 GB or more is better for Lab 4).
3. Clone the repository (Step 4), run `code .`, and click **Reopen in Container**
   when VS Code offers it (or `Cmd+Shift+P` → **Dev Containers: Reopen in
   Container**).
4. The first build takes 10-20 minutes. It is ready when the terminal shows
   `>>> Model prefetch complete.` Then everything in `labs.md` works exactly as
   written, and you can skip the "What is different" section above.

> This route has not yet been tested on Apple Silicon. If the build fails, use
> the native setup above instead.

<br>

<p align="center">
<b>(c) 2026 Tech Skills Transformations and Brent C. Laster. All rights reserved.</b>
</p>
