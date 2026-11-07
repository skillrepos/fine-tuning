"""Check the OPTIONAL Groq setup used by Lab 5 Step 6.

    python scripts/check_groq.py

Prints whether your key is set, which models it can reach, and whether the
model the lab uses actually answers. Works on Codespaces, macOS, Linux and
Windows. Never prints the key itself.
"""
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRIAGE = REPO / "common" / "triage.py"


def lab_model():
    """Read FRONTIER_MODEL from common/triage.py without loading the big training libraries."""
    m = re.search(r'^FRONTIER_MODEL\s*=\s*["\']([^"\']+)["\']',
                  TRIAGE.read_text(encoding="utf-8"), re.MULTILINE)
    return m.group(1) if m else None


def main():
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        print("GROQ_API_KEY   = MISSING")
        print()
        print("Set it first (README.md, section 'Optional: a free Groq API key'):")
        print("  Codespace / Mac / Linux / Git Bash :  source scripts/setup-key.sh")
        print("  Windows PowerShell                 :  .\\scripts\\setup-key.ps1")
        print()
        print("Remember: this key is OPTIONAL. Every lab except Lab 5 Step 6 runs without it.")
        return 1
    print(f"GROQ_API_KEY   = set ({len(key)} characters)")

    try:
        from groq import Groq
        import groq as groq_pkg
    except ImportError:
        print("The 'groq' package is not installed in this Python environment.")
        print("Activate your .venv, then:  pip install -r requirements.txt")
        return 1

    model = lab_model()
    if not model:
        print(f"Could not find FRONTIER_MODEL in {TRIAGE}")
        return 1
    print(f"Lab 5 model    = {model}   (set in common/triage.py)")
    print()

    client = Groq(max_retries=2)

    try:
        available = sorted(m.id for m in client.models.list().data)
    except groq_pkg.AuthenticationError:
        print("FAIL  Groq rejected the key. Re-copy it from console.groq.com/keys")
        print("      and run the setup script again.")
        return 1
    except groq_pkg.APIConnectionError:
        print("FAIL  Could not reach api.groq.com - check your network connection")
        print("      (a company VPN or firewall can block it).")
        return 1

    print("Chat models your key can reach right now:")
    for m in available:
        if not re.search(r"whisper|orpheus|guard|tts", m):   # skip non-chat models
            print("  " + m)
    print()

    if model not in available:
        print(f"FAIL  {model} is not available - Groq has probably retired it.")
        print("      Pick a model from the list above, put its name in FRONTIER_MODEL")
        print("      in common/triage.py, and run this check again. See also")
        print("      https://console.groq.com/docs/deprecations")
        return 1

    try:
        # This model always "thinks" before answering, and the thinking counts
        # toward the reply's length limit, so allow plenty even for a short reply.
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the word ok."}],
            max_completion_tokens=128,
        )
    except groq_pkg.RateLimitError:
        print(f"WAIT  {model} is rate-limited right now (free tier: a few requests")
        print("      per minute). Wait a minute and run this check again.")
        return 1
    except Exception as e:
        print(f"FAIL  {model} did not answer: {type(e).__name__}: {e}")
        return 1

    print(f"OK    {model} responds. Lab 5 Step 6 is ready to run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
