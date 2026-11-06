#!/usr/bin/env bash
set -euo pipefail
cd /workspace
export HOME=/workspace
export PIP_CACHE_DIR=/workspace/.pip-cache
export PYTHONPATH=/workspace/.nomad-packages
export MPLCONFIGDIR=/workspace/.mpl
python -m pip install --target "$PYTHONPATH" -r requirements-deepdish.txt
python -m pip freeze > requirements-nomad-lock.txt
python -c 'import torch; assert torch.cuda.device_count()==1; x=torch.ones(2,device="cuda"); print(torch.__version__,torch.cuda.get_device_name(0),x.sum().item())'
printf '%s\n' "${DEEPDISH_NODE:-unknown}" > deepdish_node.txt
export GITHUB_USER=Dorrie-ge
export REPO_URL=https://github.com/Dorrie-ge/mlds-diego-hw2-dqn
bash run_deepdish.sh
printf '%s\n' "Completed all training, evaluation and report generation" > RUN_COMPLETE
