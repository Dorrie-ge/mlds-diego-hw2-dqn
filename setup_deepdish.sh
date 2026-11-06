#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
.venv/bin/python -m pip install -r requirements-deepdish.txt
.venv/bin/python -m pip freeze > requirements-deepdish-lock.txt
.venv/bin/python -c 'import torch; print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_arch_list())'
touch SETUP_COMPLETE
