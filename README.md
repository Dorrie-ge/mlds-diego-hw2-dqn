# Diego - Assignment 2: Deep Q-Network

Public repository: https://github.com/Dorrie-ge/mlds-diego-hw2-dqn

This package implements both required environments. `report_DRAFT.pdf` is explicitly incomplete until the MsPacman long run, 500-episode evaluations, and deepdish GPU execution are available. Do not submit the draft as a completed assignment.

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

Request one GPU using the course's documented scheduler procedure. The allocation command is site-specific and deliberately not guessed here. Within the allocated job, from this directory:

```bash
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
python validate.py --checkpoint results/cartpole/best.pt --out results/validation.json
python plots.py
python report.py --github-user your_actual_username --repo-url https://github.com/your_actual_username/your_actual_public_repository --final
```

For a report with missing experiments visibly marked, use `python report.py` without `--final`. The final flag rejects missing metadata, evaluation counts other than 500, a mismatched environment, fewer than 4000 MsPacman training episodes, failure to improve over random in MsPacman, or absence of GPU run evidence. It does not verify that a repository is publicly accessible: manually open the real repository URL while logged out before submission.

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
- `mspacman_smoke/`: an untrained two-episode CPU integration run only, not learning/convergence evidence and excluded from report performance figures.
- `config.json`, `hardware.txt`: hyperparameters, environment/library versions and GPU provenance. The local CPU CartPole run is useful evidence but does not satisfy the specified deepdish GPU requirement.

A seed-42 run is not a multi-seed generalization claim. Avoid retuning on the reported test set. If results are inspected during tuning, use a fresh final test seed range and document that change.

## Public repository and submission

Create a public GitHub repository under your real account and upload every `.py`, `.sh`, requirements file, README, raw CSV/JSON, figures, and final report. The `.gitignore` omits large model checkpoints; share those separately or use a GitHub release if desired. Record the actual username, repository name and URL in the report. Submit the final PDF to Canvas by November 11, 2026, 11:59 pm, per the assignment. This package does not publish or submit to Canvas automatically.

## Sources

- https://ale.farama.org/environments/ms_pacman/ (v5 namespace, nine actions, skip=4 and sticky-action probability=.25)
- https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/ (bootstrap on truncation, not termination)
- https://www.nature.com/articles/nature14236 (DQN, replay and target network)

### Actual pixel-training integration check

`python pixel_smoke.py` collects 1024 real ALE transitions and performs 60 updates on a fixed 64-transition batch with a frozen target network. `results/pixel_smoke.json` records the first/last-five mean Huber losses and checks that target parameters did not change. A falling fixed-batch loss supports correct CNN/replay/optimizer integration; it is explicitly not convergence or policy-improvement evidence.

## Nomad submission for this deepdish account

The node has Nomad-managed GPU reservations. Do not infer availability solely from nvidia-smi. `deepdish.nomad.hcl` requests exactly one GPU through the scheduler and mounts the isolated assignment directory. It is specific to netid wer7892 (numeric UID/GID 1215055794), so other accounts must change its user and volume fields.

```bash
cd ~/diego_hw2_codex_20261106
nomad job validate deepdish.nomad.hcl
nomad job plan deepdish.nomad.hcl
nomad job run -detach deepdish.nomad.hcl
nomad job status wer7892-diego-hw2
```

The container uses PyTorch 2.5.1 with CUDA 11.8, an isolated dependency directory, and one scheduler-assigned GPU. `RUN_COMPLETE` is written only after all experiments and final report generation succeed. A pending Nomad allocation is not a running training experiment. Do not stop or alter other users' jobs to gain access to their reserved devices.
