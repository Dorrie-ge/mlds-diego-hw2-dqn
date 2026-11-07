# Reproducing my experiments

I keep my training/evaluation implementation at the repository root, all scientific data under `results/`, my original execution records under `artifacts/run/`, and the complete report in `report.pdf`, linked from README. I trained on deepdish1 with one GTX 1080, PyTorch 2.5.1+cu118 and Python 3.10. The completed experiment exited with status 0.

## Installation

For the recorded deepdish environment I use:

```bash
bash setup_deepdish.sh
source .venv/bin/activate
```

I pin its dependencies in `requirements-deepdish.txt` and retain its complete lock in `requirements-deepdish-lock.txt`. My original local Python 3.14 setup is separately captured in `requirements.txt` and `requirements-lock.txt`. For local figure/report generation I install those local requirements in a virtual environment. I retain these environments separately because the CUDA build used on deepdish differs from the macOS build.

The ALE package used in my run supplies ROMs. I use exactly `CartPole-v1` and `ALE/MsPacman-v5` with RGB images and nine actions.

## Times New Roman

I use actual Times New Roman TTF files for every PDF text element and every chart label. I embed the used font faces in the PDF and link that PDF from README.

I use four licensed font files (regular, bold, italic and bold italic). `report_fonts.py` searches standard macOS, Windows and Microsoft core-font directories on Linux. For another location I set:

```bash
export TNR_FONT_DIR=/path/to/my/licensed/TimesNewRoman/fonts
```

I support names such as `Times New Roman.ttf`, `Times New Roman Bold.ttf`, `Times New Roman Italic.ttf`, `Times New Roman Bold Italic.ttf`, or the Microsoft core-font names `times.ttf`, `timesbd.ttf`, `timesi.ttf`, `timesbi.ttf`. I do not redistribute these font binaries. My rendering code raises an explicit error if Times New Roman is unavailable, avoiding a silent substitute.

I regenerate the existing evidence and README index with:

```bash
python plots.py --results results
python report.py --github-user Dorrie-ge --repo-url https://github.com/Dorrie-ge/mlds-diego-hw2-dqn --final
python build_readme.py
```

These commands redraw retained evidence; they do not retrain the model or change experimental scores.

## My full single-GPU pipeline

The account owner confirmed direct SSH use of an idle GPU is permitted. I check current use before selecting a device. My recorded run used GPU 0. I provide the optional, unused Nomad alternative under `deployment/nomad/`.

```bash
export CUDA_VISIBLE_DEVICES=0
export GITHUB_USER=Dorrie-ge
export REPO_URL=https://github.com/Dorrie-ge/mlds-diego-hw2-dqn
mkdir -p artifacts/run
nohup bash run_deepdish.sh > artifacts/run/direct_training.log 2>&1 < /dev/null &
```

I create `artifacts/run/` before using the redirected log if it does not already exist. I use a file lock to prevent duplicate launches and check that exactly one GPU is visible. I write hardware, provenance, start/end timestamps and exit status under `artifacts/run/`. I preserve existing results before rerunning because the pipeline writes to the standard result directories.

My pipeline trains both games, evaluates each selected model and its random baseline for 500 episodes, runs validation, produces figures, and builds the PDF. `RUN_COMPLETE` is written only after all those steps succeed. I refresh the README index separately with `build_readme.py`.

## Individual experiments

```bash
python dqn.py train --game cartpole --episodes 3000 --out results/cartpole --device cuda --early-stop
python dqn.py train --game mspacman --episodes 6000 --out results/mspacman --device cuda --validate-every 100
python dqn.py evaluate --checkpoint results/cartpole/best.pt --episodes 500 --out results/cartpole/evaluation --device cuda
python dqn.py evaluate --checkpoint results/cartpole/best.pt --episodes 500 --random --out results/cartpole/random --device cuda
python dqn.py evaluate --checkpoint results/mspacman/best.pt --episodes 500 --out results/mspacman/evaluation --device cuda
python dqn.py evaluate --checkpoint results/mspacman/best.pt --episodes 500 --random --out results/mspacman/random --device cuda
python validate.py --checkpoint results/cartpole/best.pt --pixel-checkpoint results/mspacman/best.pt --pixel-episodes 50 --out results/validation.json
```

I keep the default ordinary DQN for my reported results. I include `--double` for separate Double DQN experiments and would save them to different directories. I use `best.pt` for evaluation and `last.pt` as the latest network/optimizer snapshot; I do not claim exact replay/RNG training-resume support.

## How I validate correctness

1. **Known Bellman solutions.** I train a deterministic two-state MDP through the same learning update as the game agents. I expect values `gamma*c` and `c` for delayed/terminal rewards, and zero for quitting. I test gamma .95, .99 and 0 and reward scales 0, 1 and 3. I require maximum Q error below .02. This checks delayed credit, reward scaling and the zero-reward control; it does not by itself prove pixel control.
2. **Termination versus truncation.** I hold reward 1 and successor value 10 fixed at gamma .95. I expect targets 1.0 for termination and 10.5 for truncation. I save the true final successor before reset.
3. **CartPole state intervention.** I add initial angle offsets -.15, -.075, 0, .075 and .15 radians to the same 100 unseen seeds per condition. I report scores and 500-step success fractions, with approximate 1.96-standard-error intervals. This probes recovery beyond the narrow reset distribution.
4. **Paired random baseline.** I evaluate each greedy policy and a uniform-random policy on the same 500 seeds. I calculate paired score differences, keeping these seeds separate from training and checkpoint selection.
5. **Non-standard calibration diagnostic.** I compare initial greedy Q with realized discounted returns, color by episode length, and plot the error against length. I use sign-clipped rewards for MsPacman value calibration and raw scores for performance. I retain censoring limitations at truncation. My figures reveal imperfect value calibration despite strong control.
6. **First ten transitions.** I retain the first ten state/action/reward/Q/termination records for each evaluation in `*_first10.json`; image traces retain pixel summaries. I inspect these to relate selected actions to observed states and rewards.
7. **Replay histories and contrast.** I use numbered frames across resets and ring overwrite to verify exact histories without borrowing frames across episodes. I check the assignment's RGB-sum-448 removal as a preprocessing intervention.
8. **Image-information dependence.** I freeze the selected MsPacman network and evaluate 50 separate paired seeds with intact history, the latest frame repeated, and a constant image. I retain raw scores in `results/mspacman/ablation.csv`, action entropy, paired intervals and checkpoint SHA-256 in `validation.json`. This tests inference dependence, not equal-budget retraining. My report records that both removals reduce scores in this run.

I run these tests with `validate.py` using the command above. If image-intervention evidence is absent or its checkpoint hash changes, `report.py` can generate it from the selected checkpoint on CPU before producing a complete report. I do not allocate a second GPU for these inference checks.

## Reading my evidence

- `results/*/train.csv`: raw episode reward, epsilon, visited/fixed-probe Q summaries, loss, validation score and wall time.
- `results/*/evaluation.csv/json`: 500 greedy test scores, lengths, initial Q, discounted return, termination/truncation, and summary statistics.
- `results/*/random.csv/json`: the matched-seed uniform-random baseline.
- `results/validation.json`: analytic errors, robustness results, image interventions and selected checkpoint identity.
- `artifacts/run/`: my original GPU execution log, hardware and provenance. Reorganization preserves their contents.
- `SHA256SUMS.json`: retained file hashes. Model hashes refer to the downloadable package; models are excluded from GitHub.

I use training seed 42, final test seeds 100000-100499, validation seeds starting at 200000, and diagnostic image-intervention seeds 500000-500049. I do not claim multi-seed training robustness. I describe additional validation I would perform over three months in the report.
