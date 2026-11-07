"""Regenerate every plot from raw CSV; no synthesized learning data."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_fonts import configure_plot_fonts
configure_plot_fonts()
plt.rcParams.update({'figure.dpi':150,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})


def read(path):
    with open(path) as f: return list(csv.DictReader(f))


def plots(root):
    root=Path(root)
    for game in ['cartpole','mspacman']:
        d=root/game
        if not (d/'train.csv').exists(): continue
        rows=read(d/'train.csv'); ep=np.array([int(r['episode']) for r in rows]); rewards=np.array([float(r['reward']) for r in rows])
        fig,ax=plt.subplots(figsize=(7,3.6))
        for key,label in [('visited_mean_max_q','Visited states: mean max Q'),('fixed_mean_max_q','Fixed probe: mean max Q'),('fixed_peak_max_q','Fixed probe: largest max Q')]:
            ax.plot(ep,[float(r[key]) for r in rows],label=label,linewidth=1)
        ax.set(xlabel='Training episode',ylabel='Q value (training reward units)',title=f'{game}: maximum action values'); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(d/'q_values.png'); plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,3.6)); ax.plot(ep,rewards,alpha=.4,label='Raw episode reward')
        if len(rewards)>=100: ax.plot(ep[99:],np.convolve(rewards,np.ones(100)/100,'valid'),label='Trailing 100-episode mean',color='#bd4a27')
        ax.set(xlabel='Training episode',ylabel='Undiscounted game score',title=f'{game}: training rewards'); ax.legend(); fig.tight_layout(); fig.savefig(d/'rewards.png'); plt.close(fig)
        if not (d/'evaluation.csv').exists(): continue
        ev=read(d/'evaluation.csv'); scores=np.array([float(r['reward']) for r in ev]); stats=json.loads((d/'evaluation.json').read_text())
        fig,ax=plt.subplots(figsize=(7,3.6)); ax.hist(scores,bins=25,edgecolor='white'); ax.axvline(scores.mean(),color='#bd4a27',label=f'Mean {scores.mean():.2f}; sample SD {scores.std(ddof=1):.2f}')
        ax.set(xlabel='Undiscounted game score',ylabel='Number of episodes',title=f'{game}: {len(scores)} held-out greedy episodes'); ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(d/'histogram.png'); plt.close(fig)
        q=np.array([float(r['q_start']) for r in ev]); mc=np.array([float(r['discounted_return']) for r in ev]); lengths=np.array([int(r['length']) for r in ev])
        fig,axs=plt.subplots(1,2,figsize=(9,3.7))
        sc=axs[0].scatter(mc,q,c=lengths,cmap='viridis',s=14,alpha=.5)
        lo=min(mc.min(),q.min()); hi=max(mc.max(),q.max()); axs[0].plot([lo,hi],[lo,hi],'--',color='gray')
        axs[0].set(xlabel='Realized discounted return',ylabel='Q(initial state, greedy action)',title='Calibration at episode start')
        fig.colorbar(sc,ax=axs[0],label='Episode length')
        axs[1].scatter(lengths,q-mc,s=12,alpha=.4); axs[1].axhline(0,color='gray',linestyle='--')
        axs[1].set(xlabel='Episode length',ylabel='Q minus realized return',title='Error versus survival / game length')
        fig.suptitle(f'{game}: non-standard value calibration diagnostic'); fig.tight_layout(); fig.savefig(d/'calibration.png'); plt.close(fig)
    v=root/'validation.json'
    if v.exists():
        data=json.loads(v.read_text())
        if 'cartpole_angle_perturbation' in data:
            rows=data['cartpole_angle_perturbation']; fig,ax=plt.subplots(figsize=(7,3.4))
            means=np.array([r['mean'] for r in rows]); se=np.array([r['std']/np.sqrt(r['n']) for r in rows])
            ax.errorbar([r['angle_offset_rad'] for r in rows],means,yerr=1.96*se,fmt='o-',capsize=4)
            ax.set(xlabel='Added initial pole angle (radians)',ylabel='Mean episode reward (approx. 95% interval)',title='Held-out initial-state robustness: 100 paired seeds per offset')
            fig.tight_layout(); fig.savefig(root/'cartpole'/'robustness.png'); plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--results',default='results'); a=p.parse_args(); plots(a.results)
