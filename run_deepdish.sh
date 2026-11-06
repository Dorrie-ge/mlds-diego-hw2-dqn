#!/usr/bin/env bash
# Run INSIDE an allocation with one GPU. Ask your course's scheduler for one GPU first.
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
export MPLCONFIGDIR="${PWD}/.mpl"
python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(torch.cuda.get_device_name(0))'
nvidia-smi > hardware.txt
python dqn.py train --game cartpole --episodes 3000 --out results/cartpole --device cuda --early-stop
python dqn.py train --game mspacman --episodes 6000 --out results/mspacman --device cuda --validate-every 100
for game in cartpole mspacman; do
  python dqn.py evaluate --checkpoint "results/$game/best.pt" --episodes 500 --out "results/$game/evaluation" --device cuda
  python dqn.py evaluate --checkpoint "results/$game/best.pt" --episodes 500 --random --out "results/$game/random" --device cuda
done
python validate.py --checkpoint results/cartpole/best.pt --out results/validation.json
python plots.py
# Requires actual public repository metadata; refuses to mark an incomplete run final.
python report.py --github-user "${GITHUB_USER:?Set GITHUB_USER}" --repo-url "${REPO_URL:?Set REPO_URL}" --final
