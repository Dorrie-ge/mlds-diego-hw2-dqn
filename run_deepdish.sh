#!/usr/bin/env bash
# Use exactly one GPU. Direct SSH execution is course-authorized for this account.
set -euo pipefail
cd "$(dirname "$0")"
# Prevent duplicate direct/Nomad launches of this pipeline.
exec 9>.training.lock
flock -n 9 || { echo "A training pipeline is already running"; exit 1; }
trap 'status=$?; printf "%s\n" "$status" > run_exit_status.txt' EXIT
if [[ -z "${VIRTUAL_ENV:-}" && -x .venv/bin/python ]]; then
  source .venv/bin/activate
fi
date -u +%Y-%m-%dT%H:%M:%SZ > RUN_STARTED
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
export MPLCONFIGDIR="${PWD}/.mpl"
python -u -c 'import torch; from dqn import QNet; assert torch.cuda.is_available() and torch.cuda.device_count()==1; model=QNet("mspacman",9).cuda(); q=model(torch.zeros((2,4,88,80),device="cuda")); q.square().mean().backward(); torch.cuda.synchronize(); print("GPU preflight passed:",torch.__version__,torch.cuda.get_device_name(0),flush=True)'
nvidia-smi > hardware.txt
python -u dqn.py train --game cartpole --episodes 3000 --out results/cartpole --device cuda --early-stop
python -u dqn.py train --game mspacman --episodes 6000 --out results/mspacman --device cuda --validate-every 100
for game in cartpole mspacman; do
  python -u dqn.py evaluate --checkpoint "results/$game/best.pt" --episodes 500 --out "results/$game/evaluation" --device cuda
  python -u dqn.py evaluate --checkpoint "results/$game/best.pt" --episodes 500 --random --out "results/$game/random" --device cuda
done
python validate.py --checkpoint results/cartpole/best.pt --out results/validation.json
python plots.py
# Requires actual public repository metadata; refuses to mark an incomplete run final.
python report.py --github-user "${GITHUB_USER:?Set GITHUB_USER}" --repo-url "${REPO_URL:?Set REPO_URL}" --final

date -u +%Y-%m-%dT%H:%M:%SZ > RUN_COMPLETE
