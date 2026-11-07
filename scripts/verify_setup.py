"""Check that this machine can run the workshop labs - before the workshop.

    python scripts/verify_setup.py

Run it from the repository folder with your .venv activated. It takes under a
minute and needs no internet connection. It checks:

  1. Python version and the installed libraries
  2. That the workshop models have been downloaded
  3. That 4-bit QLoRA training (Lab 4) actually works on THIS machine, using a
     tiny made-up model - the part most likely to differ between machines
  4. Whether VS Code's `code -d` diff command is available
  5. Whether the optional Groq key is set

Works on GitHub Codespaces, macOS, Windows and Linux.
"""
import importlib
import os
import platform
import shutil
import sys
import tempfile
import time

# Keep the output readable: hide library warnings and progress bars, the same
# way the labs do.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common.quiet  # noqa: E402,F401

PASS, WARN, FAIL = "OK  ", "WARN", "FAIL"
results = []


def report(status, label, detail=""):
    results.append(status)
    line = f"  [{status}] {label}"
    if detail:
        line += f"  -  {detail}"
    print(line)


def header(text):
    print()
    print(text)
    print("-" * len(text))


# ---------------------------------------------------------------------------
header("1. Python and libraries")

machine = platform.machine().lower()
system = platform.system()
print(f"  {system} {platform.release()} on {machine}, Python {platform.python_version()}")

if system == "Darwin" and machine in ("x86_64", "amd64"):
    report(FAIL, "Intel Mac",
           "PyTorch and bitsandbytes no longer publish Intel-Mac builds. "
           "Please use the GitHub Codespace instead (see README.md).")
if sys.version_info < (3, 10):
    report(FAIL, "Python version", "3.10 or newer is required; 3.11 is recommended")
elif sys.version_info >= (3, 14):
    report(WARN, "Python version", "3.11 is what the labs are tested on; very new "
           "Pythons sometimes lack library builds")
else:
    report(PASS, "Python version")

LIBS = ["torch", "transformers", "peft", "trl", "datasets", "accelerate",
        "bitsandbytes", "lm_eval", "sklearn", "groq"]
missing = []
for name in LIBS:
    try:
        mod = importlib.import_module(name)
        report(PASS, f"{name:<13}", getattr(mod, "__version__", "installed"))
    except Exception as e:
        missing.append(name)
        report(FAIL, f"{name:<13}", f"cannot import ({type(e).__name__}: {e})")
if missing:
    print("\n  Fix: activate .venv, then  pip install -r requirements.txt")

# ---------------------------------------------------------------------------
header("2. Workshop models downloaded")

MODELS = ["distilgpt2", "HuggingFaceTB/SmolLM2-360M-Instruct",
          "TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
try:
    from huggingface_hub import snapshot_download
    for m in MODELS:
        try:
            snapshot_download(m, local_files_only=True)
            report(PASS, m)
        except Exception:
            report(FAIL, m, "not downloaded yet - run: python scripts/prefetch_models.py")
except ImportError:
    report(FAIL, "huggingface_hub", "not installed - pip install -r requirements.txt")

# ---------------------------------------------------------------------------
header("3. 4-bit QLoRA training on this machine (Lab 4)")

if "torch" in missing or "transformers" in missing or "peft" in missing \
        or "bitsandbytes" in missing:
    report(FAIL, "4-bit test", "skipped - libraries above are missing")
else:
    import torch
    from transformers import (AutoModelForCausalLM, BitsAndBytesConfig,
                              LlamaConfig, LlamaForCausalLM)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers.utils import logging as hf_logging
    hf_logging.disable_progress_bar()

    tmp = tempfile.mkdtemp(prefix="ft-verify-")
    try:
        # A tiny made-up model built the same way as SmolLM2, so the test
        # needs no download.
        torch.manual_seed(0)
        cfg = LlamaConfig(vocab_size=512, hidden_size=256, intermediate_size=512,
                          num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=4, max_position_embeddings=128)
        LlamaForCausalLM(cfg).save_pretrained(tmp)

        t0 = time.time()
        # Exactly the configuration Lab 4 uses.
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            tmp, quantization_config=bnb, dtype=torch.bfloat16,
            device_map={"": "cpu"})
        n4 = sum(1 for m in model.modules() if type(m).__name__ == "Linear4bit")
        if n4 == 0:
            raise RuntimeError("model loaded, but no layer was stored in 4 bits")
        report(PASS, "load in 4 bits", f"{n4} layers stored in 4 bits")

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        model = get_peft_model(model, LoraConfig(
            r=8, lora_alpha=16, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
        ids = torch.randint(0, 512, (2, 32))
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        grads = [p.grad for n, p in model.named_parameters()
                 if "lora_" in n and p.grad is not None]
        if not grads or all(float(g.abs().sum()) == 0.0 for g in grads):
            raise RuntimeError("training did not reach the LoRA adapters")
        report(PASS, "train through 4-bit base",
               f"training reached {len(grads)} adapter pieces "
               f"({time.time() - t0:.1f}s)")
    except Exception as e:
        report(FAIL, "4-bit test", f"{type(e).__name__}: {e}")
        print("\n  Lab 4 will not run on this machine. Labs 1-3 and 5 are unaffected.")
        print("  Options: use the GitHub Codespace for Lab 4, or follow along with")
        print("  the instructor's screen for that lab.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------------------------------
header("4. Diff/merge tool for the lab steps")

if shutil.which("code"):
    report(PASS, "`code` command", "the `code -d` steps in labs.md will work as written")
else:
    report(WARN, "`code` command not found",
           "install VS Code and add `code` to your PATH, or use another diff "
           "tool - see your Local_Setup guide, section 'Merging the lab code'")

# ---------------------------------------------------------------------------
header("5. Optional: Groq key for Lab 5 Step 6")

if os.environ.get("GROQ_API_KEY", "").strip():
    report(PASS, "GROQ_API_KEY", "set - run  python scripts/check_groq.py  to test it")
else:
    report(WARN, "GROQ_API_KEY not set",
           "fine - only the optional Lab 5 Step 6 needs it (see README.md)")

# ---------------------------------------------------------------------------
print()
if FAIL in results:
    print("RESULT: something needs fixing - see the FAIL lines above.")
    sys.exit(1)
print("RESULT: ready for the workshop." +
      (" (WARN lines are optional.)" if WARN in results else ""))
