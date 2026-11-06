"""
Turn off messages from the libraries that are not meant for you.

The Hugging Face and 4-bit libraries print progress bars, internal warnings
and notes for their own developers. None of that is part of what the labs
teach, so every lab imports this file first, before those libraries load.
Real errors still appear.
"""

import logging
import os
import warnings

# Set before the libraries are imported, so they start up quiet.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

for name in ("transformers", "datasets", "peft", "bitsandbytes",
             "accelerate", "huggingface_hub", "torch"):
    logging.getLogger(name).setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# In case the libraries were already imported, switch them off directly too.
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
except Exception:
    pass

try:
    import datasets
    datasets.disable_progress_bars()
    datasets.logging.set_verbosity_error()
except Exception:
    pass
