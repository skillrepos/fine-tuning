#!/usr/bin/env bash
# Create the Python environment for the fine-tuning workshop.
#   usage: bash scripts/pysetup.sh .venv
set -e

PYTHON_ENV=${1:-.venv}

python3 -m venv "./$PYTHON_ENV"
export PATH="./$PYTHON_ENV/bin:$PATH"
grep -qxF "source $(pwd)/$PYTHON_ENV/bin/activate" ~/.bashrc \
    || echo "source $(pwd)/$PYTHON_ENV/bin/activate" >> ~/.bashrc
source "./$PYTHON_ENV/bin/activate"

pip3 install --upgrade pip

# CPU-only PyTorch. Without the explicit index, pip pulls ~3 GB of CUDA
# libraries that are dead weight on a Codespace.
echo ">>> Installing CPU-only PyTorch (this is the slow one, ~2 minutes)"
pip3 install "torch>=2.9" --index-url https://download.pytorch.org/whl/cpu

echo ">>> Installing the rest of the stack"
pip3 install -r ./requirements.txt

echo ">>> Versions installed:"
python3 - <<'PY'
import torch, transformers, peft, trl, datasets, bitsandbytes
print(f"  torch        {torch.__version__}")
print(f"  transformers {transformers.__version__}")
print(f"  peft         {peft.__version__}")
print(f"  trl          {trl.__version__}")
print(f"  datasets     {datasets.__version__}")
print(f"  bitsandbytes {bitsandbytes.__version__}")
PY
echo ">>> Python environment ready."
