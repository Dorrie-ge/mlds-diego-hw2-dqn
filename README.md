# Diego - Assignment 2: Deep Q-Network

Public repository: https://github.com/Dorrie-ge/mlds-diego-hw2-dqn

This package contains completed single-GPU deepdish experiments and the final `report.pdf`.

- CartPole-v1: 1675 training episodes; 500/500 held-out episodes reached 500 steps (mean 500, sample SD 0).
- ALE/MsPacman-v5: 6000 training episodes; 500 held-out episodes averaged 2752.08 raw points (sample SD 798.01), versus 256.88 for the matched-seed random baseline.
- Final image-history interventions, analytic MDP experiments, initial-state robustness checks, raw logs, figures, run provenance and the selected trained checkpoints are included. Models are omitted from GitHub via `.gitignore` but included in the downloadable final package.
- The report retains limitations: one training seed, value-calibration error, and inference interventions rather than retraining ablations. No submission to Canvas has been made.

## Installation

Tested locally with Python 3.14 on macOS; deepdish requires a matching CUDA-enabled PyTorch build. Start inside a virtual environment. The versions in `requirements.txt` are the versions actually tested; `requirements-lock.txt` captures all local transitive dependencies. On a cluster, use the PyTorch CUDA wheel compatible with its driver rather than a macOS wheel.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# For deepdish Python 3.10 / GTX 1080, use setup_deepdish.sh instead.
```

The tested ALE package supplies its ROMs. If a cluster installation cannot create `ALE/MsPacman-v5`, follow its installed ALE version's official ROM instructions. No substitution of a v4 environment or RAM observations is permitted.

## Full experiment on deepdish, one GPU

The account owner confirmed that direct SSH execution on an idle GPU is allowed. On deepdish, select one currently idle GPU and run from this directory (the active run uses GPU 0):

```bash
export CUDA_VISIBLE_DEVICES=0
export GITHUB_USER=your_actual_username
export REPO_URL=https://github.com/your_actual_username/your_actual_public_repository
bash run_deepdish.sh
```

This runs CartPole training, 6000 MsPacman episodes, held-out greedy and uniform-random evaluations (500 episodes each), behavioral validation, figures, and the PDF report. Check convergence rather than assuming 6000 episodes suffice. The default algorithm is ordinary target-network DQN; `--double` enables a documented Double DQN variant for a separate ablation run. Preserve each ablation in a different `--out` directory.

## Individual commands

```bash
python dqn.py train --game cartpole --episodes 3000 --out results/cartpole --device cuda --early-stop
python dqn.py train --game mspacman --episodes 6000 --out results/mspacman --device cuda --validate-every 100
python dqn.py evaluate --checkpoint results/cartpole/best.pt --episodes 500 --out results/cartpole/evaluation --device cuda
python dqn.py evaluate --checkpoint results/cartpole/best.pt --episodes 500 --random --out results/cartpole/random --device cuda
python dqn.py evaluate --checkpoint results/mspacman/best.pt --episodes 500 --out results/mspacman/evaluation --device cuda
python dqn.py evaluate --checkpoint results/mspacman/best.pt --episodes 500 --random --out results/mspacman/random --device cuda
python validate.py --checkpoint results/cartpole/best.pt --pixel-checkpoint results/mspacman/best.pt --pixel-episodes 50 --out results/validation.json
python plots.py
python report.py --github-user your_actual_username --repo-url https://github.com/your_actual_username/your_actual_public_repository --final
```

Use `python report.py --github-user Dorrie-ge --repo-url https://github.com/Dorrie-ge/mlds-diego-hw2-dqn --final` to regenerate the completed report from retained data. For intentionally incomplete experiments, omit `--final` to produce a visibly marked draft. The final flag rejects missing metadata, evaluation counts other than 500, a mismatched environment, fewer than 4000 MsPacman training episodes, failure to improve over random in MsPacman, or absence of GPU run evidence. It does not verify that a repository is publicly accessible: manually open the real repository URL while logged out before submission.

## What the validation experiments mean

1. `validate.py` learns a deterministic two-state MDP through the same Bellman target, Huber loss, gradient clipping, target synchronization, and optimizer update as the game agent. Exact optimal Q values are known: Q(s0,a0)=gamma*reward_scale, Q(s1,a0)=reward_scale, quit actions=0. Tests at gamma=.95, .99, 0 and reward scales 0, 1, 3 must have maximum absolute error below .02. This detects broken discounting, reward propagation, terminal masking, and fabricated value growth with zero rewards. It tests the learning rule; the toy linear model cannot prove that a pixel CNN learns the game.
2. A controlled terminal/time-limit pair has the same reward 1 and successor value 10. At gamma=.95 its targets must be 1 and 10.5 respectively. This verifies the semantics that otherwise silently bias long survival episodes.
3. CartPole is evaluated on the same 100 unseen seeds under initial pole-angle offsets -.15, -.075, 0, .075, .15 radians. Returns and 500-step success rates reveal robustness away from the narrow reset distribution. The intervals plotted are approximate 1.96 standard errors; seeds are paired across conditions.
4. Uniform-random policies use the exact evaluation seeds of greedy policies. Both raw CSVs are retained, enabling paired differences rather than comparing unrelated games. No 500-episode test seeds are used for checkpoint selection.
5. Non-standard plots compare initial greedy Q with independently realized discounted return, colored by episode length, plus calibration error versus episode length. MsPacman uses clipped rewards in this diagnostic to match training units, while all performance statistics use raw scores. Truncated returns are finite observed prefixes; Q bootstraps past those cutoffs, so their gap is not automatically overestimation. CartPole's gamma=.95 yields a saturation value of 20; large differences in survival above 100 steps may barely change Q.
6. Both evaluation commands retain the first ten state/action/reward/Q/termination transitions as JSON. For image inputs the trace stores pixel summaries, not thousands of numbers. Inspect it alongside the reset and update logic.
7. The replay-history intervention uses two short episodes and a wrapped ring with uniquely numbered frames; it verifies that histories are reset and never borrow frames from a prior episode. A contrast pixel checks the assignment's background-removal transformation. These support the behavioral tests but are not substitutes for them.

## Files and interpretation

- `train.csv`: per-episode raw score, epsilon, visited-state mean max Q, fixed-probe mean/peak max Q, Huber loss, separate validation score, wall time.
- `best.pt`: selected only by validation seeds 200000 onward. `last.pt`: latest network, target, optimizer and counters. Training restarts from scratch; these are evaluation checkpoints, not exact replay/RNG resume snapshots.
- `evaluation.csv/json`: 500 independent test episodes, raw reward, episode length, initial Q, discounted training-unit return, sample standard deviation, and CartPole success fraction. Seeds begin at 100000 and are separate from training and model selection.
- `random.csv/json`: matched-seed baseline.
- `config.json`, `hardware.txt`, `run_provenance.json`: actual deepdish CUDA hyperparameters, library versions, host and one-GPU visibility. Final result directories contain the GPU runs; earlier CPU-only results are not mixed into their statistics.

A seed-42 run is not a multi-seed generalization claim. Avoid retuning on the reported test set. If results are inspected during tuning, use a fresh final test seed range and document that change.

## Public repository and submission

Create a public GitHub repository under your real account and upload every `.py`, `.sh`, requirements file, README, raw CSV/JSON, figures, and final report. The `.gitignore` omits large model checkpoints; share those separately or use a GitHub release if desired. Record the actual username, repository name and URL in the report. Submit the final PDF to Canvas by November 11, 2026, 11:59 pm, per the assignment. This package does not publish or submit to Canvas automatically.

## Sources

- https://ale.farama.org/environments/ms_pacman/ (v5 namespace, nine actions, skip=4 and sticky-action probability=.25)
- https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/ (bootstrap on truncation, not termination)
- https://www.nature.com/articles/nature14236 (DQN, replay and target network)

### Actual pixel-training integration check

`python pixel_smoke.py` collects 1024 real ALE transitions and performs 60 updates on a fixed 64-transition batch with a frozen target network. `results/pixel_smoke.json` records the first/last-five mean Huber losses and checks that target parameters did not change. A falling fixed-batch loss supports correct CNN/replay/optimizer integration; it is explicitly not convergence or policy-improvement evidence.

## Direct SSH run for this account

The account owner confirmed direct GPU use is permitted. The earlier Nomad queue job `wer7892-diego-hw2` was stopped before starting direct execution, preventing duplicate training. The active pipeline uses exactly one physical GPU through `CUDA_VISIBLE_DEVICES=0`, an isolated `.venv`, a file lock, and a detached process. The CUDA preflight executes CNN forward/backward operations before the long run.

```bash
cd ~/diego_hw2_codex_20261106
export CUDA_VISIBLE_DEVICES=0
export GITHUB_USER=Dorrie-ge
export REPO_URL=https://github.com/Dorrie-ge/mlds-diego-hw2-dqn
nohup bash run_deepdish.sh > direct_training.log 2>&1 < /dev/null &
tail -n 20 direct_training.log
```

Check current GPU usage before choosing a device. `RUN_STARTED` records launch time; `run_exit_status.txt` records eventual exit status, and `RUN_COMPLETE` is written only after all training, evaluations and final report generation succeed. The retained completed run exited with status 0 and has a RUN_COMPLETE timestamp. A process launch alone would not establish convergence. The retained Nomad configuration is an optional scheduling alternative, not a requirement for the direct run. Do not start both pipelines.

### Final frozen-model image interventions

After the full training and 500-episode evaluations, `report.py --final` ensures that the selected MsPacman checkpoint has paired pixel-intervention evidence. If missing, it runs 50 independent diagnostic seeds on CPU for intact frame history, repeated latest frame, and a constant image. It records raw scores in `results/mspacman/ablation.csv`, action entropy, paired differences, approximate intervals and checkpoint SHA-256 in `validation.json`. This changes inference inputs only; it is not retraining and does not use those seeds for checkpoint selection. CPU evaluation does not allocate an additional GPU. The CLI command above reproduces the experiments explicitly.
