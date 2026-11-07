"""Embed every report page in README while preserving its Times New Roman layout."""
import json,os,shutil,subprocess,tempfile
from pathlib import Path
BASE=Path(__file__).resolve().parent


def main():
    exe=os.environ.get('PDFTOPPM_BIN') or shutil.which('pdftoppm')
    if not exe:raise SystemExit('Install Poppler (pdftoppm), or set PDFTOPPM_BIN to its executable.')
    out=BASE/'docs'/'report-pages';out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='report-render-') as temp:
        prefix=Path(temp)/'page'
        subprocess.run([exe,'-r','120','-png',str(BASE/'report.pdf'),str(prefix)],check=True)
        pages=sorted(Path(temp).glob('page-*.png'),key=lambda p:int(p.stem.rsplit('-',1)[1]))
        if not pages:raise RuntimeError('PDF rendering produced no pages')
        for old in out.glob('page-*.png'):old.unlink()
        for i,p in enumerate(pages,1):shutil.copy2(p,out/f'page-{i:02d}.png')
        count=len(pages)
    cp=json.loads((BASE/'results/cartpole/evaluation.json').read_text());ms=json.loads((BASE/'results/mspacman/evaluation.json').read_text());rnd=json.loads((BASE/'results/mspacman/random.json').read_text())
    content=f'''# MLDS Assignment 2 - Deep Q-Network

I implemented and trained DQN for **CartPole-v1** and image-based **ALE/MsPacman-v5** on a single deepdish GPU. I present my complete first-person report below, with the same Times New Roman typography as the PDF.

**[Read or download the final PDF](report.pdf)** · **[Reproduce my experiments](docs/REPRODUCING.md)** · **[Browse raw results](results/)**

## My results

| My greedy policy | Held-out episodes | Mean raw reward | Sample SD |
| --- | ---: | ---: | ---: |
| CartPole-v1 | {cp['n']} | {cp['mean']:.2f} | {cp['std']:.2f} |
| ALE/MsPacman-v5 | {ms['n']} | {ms['mean']:.2f} | {ms['std']:.2f} |

I reached 500 steps in every CartPole test episode. For MsPacman, I completed 6000 training episodes and compared my frozen policy with a matched-seed random baseline (mean {rnd['mean']:.2f}). I also ran analytic learning controls, initial-state robustness checks and paired image-information interventions. I discuss my single-seed and calibration limitations in the report.

## Repository guide

| Location | What I keep there |
| --- | --- |
| [report.pdf](report.pdf) | My complete submission report |
| [dqn.py](dqn.py) | My networks, replay buffer, training and evaluation |
| [validate.py](validate.py) | My analytic, robustness and image-intervention experiments |
| [plots.py](plots.py), [report.py](report.py), [report_fonts.py](report_fonts.py) | Reproducible Times New Roman figures and report |
| [build_readme.py](build_readme.py) | PDF page rendering and this README |
| [results/cartpole/](results/cartpole/), [results/mspacman/](results/mspacman/) | Raw data, configurations, traces and figures |
| [artifacts/run/](artifacts/run/) | My original GPU log, hardware, provenance and completion records |
| [docs/REPRODUCING.md](docs/REPRODUCING.md) | Installation, commands and validation documentation |
| [deployment/nomad/](deployment/nomad/) | My optional Nomad configuration; I used direct SSH for the completed run |

I keep trained checkpoints out of GitHub. I retain their hashes in SHA256SUMS.json and the selected MsPacman checkpoint hash in results/validation.json.

## Regenerate my report

After installing the documented dependencies and licensed Times New Roman fonts, I regenerate the figures, report and embedded pages with:

```bash
python plots.py --results results
python report.py --github-user Dorrie-ge --repo-url https://github.com/Dorrie-ge/mlds-diego-hw2-dqn --final
python build_readme.py
```

I can regenerate the report from the retained results without retraining the networks. I use Poppler's `pdftoppm` to build the page previews. I do not redistribute font binaries; [the font setup instructions](docs/REPRODUCING.md#times-new-roman) explain the supported locations and `TNR_FONT_DIR`.

## My complete report

I embed all {count} pages directly below so the report keeps its Times New Roman font, figures, tables and pagination inside GitHub. Click a page to enlarge it, or use the PDF link above.

'''
    content+='\n\n'.join(f'![My report, page {i} of {count}](docs/report-pages/page-{i:02d}.png)' for i in range(1,count+1))+'\n'
    (BASE/'README.md').write_text(content)
    print(f'Embedded {count} report pages in README.md')

if __name__=='__main__':main()
