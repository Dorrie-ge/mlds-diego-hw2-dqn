"""Build an evidence-backed report; --final refuses incomplete experiments."""
import argparse,csv,json
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,Preformatted
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter

BASE=Path(__file__).resolve().parent

def read_csv(path):
    with open(path) as f:return list(csv.DictReader(f))


def build(a):
    root=BASE/'results'; missing=[]
    if not a.github_user or not a.repo_url: missing.append('Actual GitHub username, repository name and public repository URL have not been supplied.')
    if not (BASE/'hardware.txt').exists():missing.append('No deepdish single-GPU execution record is present. Included CartPole results were run locally on CPU.')
    for game in ['cartpole','mspacman']:
        d=root/game
        for name in ['train.csv','evaluation.json','random.json','calibration.png','q_values.png','rewards.png','histogram.png']:
            if not (d/name).exists():missing.append(f'{game}: missing {name}.')
        if (d/'evaluation.json').exists() and json.loads((d/'evaluation.json').read_text())['n']!=500:missing.append(f'{game}: evaluation count is not 500.')
        if (d/'random.json').exists() and json.loads((d/'random.json').read_text())['n']!=500:missing.append(f'{game}: random baseline count is not 500.')
        if (d/'config.json').exists():
            cfg=json.loads((d/'config.json').read_text()); expected='CartPole-v1' if game=='cartpole' else 'ALE/MsPacman-v5'
            if cfg['environment']!=expected:missing.append(f'{game}: wrong environment version.')
        if game=='mspacman' and (d/'train.csv').exists():
            rows=read_csv(d/'train.csv')
            if len(rows)<4000:missing.append('MsPacman: fewer than 4000 training episodes; inspect the learning curve and extend training.')
            if (d/'evaluation.json').exists() and (d/'random.json').exists():
                if json.loads((d/'evaluation.json').read_text())['mean']<=json.loads((d/'random.json').read_text())['mean']:missing.append('MsPacman does not outperform its matched random baseline.')
    # When the scheduled full run reaches reporting, run the final frozen-model
    # image interventions on CPU. This does not interrupt or use another GPU.
    vp=root/'validation.json'
    validation=json.loads(vp.read_text()) if vp.exists() else {}
    if a.final and not missing:
        import hashlib
        ckpt=root/'mspacman'/'best.pt'
        old=validation.get('mspacman_pixel_interventions',{})
        expected=hashlib.sha256(ckpt.read_bytes()).hexdigest() if ckpt.exists() else old.get('checkpoint_sha256')
        if not old or old.get('checkpoint_sha256')!=expected:
            if not ckpt.exists(): raise SystemExit('Final pixel validation requires the trained MsPacman checkpoint.')
            from validate import pixel_ablation
            print('Running paired frozen-model pixel interventions on CPU...',flush=True)
            validation['mspacman_pixel_interventions']=pixel_ablation(ckpt,root/'mspacman'/'ablation.csv',50)
            vp.write_text(json.dumps(validation,indent=2))
    if not validation.get('analytic_learning') or not validation.get('cartpole_angle_perturbation'):
        missing.append('Analytic learning and CartPole robustness evidence are missing.')
    if not validation.get('mspacman_pixel_interventions'):
        missing.append('Frozen-model MsPacman pixel interventions remain pending.')
    if a.final and missing:raise SystemExit('Final report refused:\n'+'\n'.join(missing))
    draft=bool(missing); output=BASE/('report_DRAFT.pdf' if draft else 'report.pdf')
    styles=getSampleStyleSheet(); styles.add(ParagraphStyle(name='Text',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=8)); styles.add(ParagraphStyle(name='SmallText',fontSize=8,leading=11,spaceAfter=6)); styles.add(ParagraphStyle(name='CodeSmall',fontName='Courier',fontSize=7,leading=10)); styles['Title'].textColor=colors.HexColor('#16384b'); styles['Heading1'].fontSize=17; styles['Heading2'].fontSize=12
    story=[]
    def p(text,style='Text'):story.append(Paragraph(text,styles[style]))
    def title(text):p(text,'Heading1')
    def heading(text):p(text,'Heading2')
    def page():story.append(PageBreak())
    def table(rows,widths=None):
        t=Table([[Paragraph(escape(str(x)),styles['SmallText']) for x in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f0f3')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#93a8b2')),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7)])); story.append(t); story.append(Spacer(1,8))
    def image(path,width=480):
        if path.exists():
            from PIL import Image as PILImage
            with PILImage.open(path) as im:w,h=im.size
            story.append(Image(str(path),width=width,height=width*h/w)); story.append(Spacer(1,5))
    title('Assignment 2 - Deep Q-Network')
    p('Diego HW2 | MLDS | Due November 11, 2026, 11:59 pm','Heading2')
    p('Evidence status: '+('INCOMPLETE DRAFT - NOT READY TO SUBMIT' if draft else 'Completed experiment report'),'Heading2')
    p('GitHub username: '+escape(a.github_user or 'NOT PROVIDED'))
    p('Repository: '+escape(a.repo_url.rstrip('/').split('/')[-1] if a.repo_url else 'NOT PROVIDED'))
    p('Public source URL: '+escape(a.repo_url or 'NOT PROVIDED'))
    heading('Scope and evidence')
    p('The implementation uses a neural action-value function, uniform experience replay and a separately synchronized target network in both games. CartPole uses gamma = 0.95; image-based MsPacman uses gamma = 0.99 and nine actions. All numbers below are computed from retained experiment logs; unexecuted experiments are explicitly identified.')
    if missing:
        heading('Requirements still outstanding')
        # Compact repeated missing filenames into a readable statement.
        for line in missing:
            if line.startswith('mspacman: missing'):continue
            p(escape(line),'SmallText')
        if any(x.startswith('mspacman: missing') for x in missing):p('MsPacman full training, its required learning curves, and its 500-episode trained-policy/random evaluations have not been completed. The two-episode smoke run does not establish learning or performance.','SmallText')
    heading('Reproduction and audit trail')
    p('Use setup_deepdish.sh for the Python 3.10 / CUDA 11.8 environment, or the local requirements.txt for the original CPU setup. On deepdish, select exactly one idle GPU with CUDA_VISIBLE_DEVICES and run bash run_deepdish.sh with GITHUB_USER and REPO_URL set. The account owner confirmed direct SSH GPU execution is permitted. README.md gives every individual training, evaluation, plotting and validation command. train.csv, evaluation.csv, random.csv, config.json, first-ten-step traces and validation.json provide the numerical audit trail. Checkpoints are selected on validation seeds, never on the 500 final test episodes.')
    provenance=BASE/'run_provenance.json'
    if provenance.exists():
        info=json.loads(provenance.read_text())
        p('Execution record: '+escape(str(info.get('hostname','deepdish')))+'; '+escape(str(info.get('gpu_name','GPU')))+'; PyTorch '+escape(str(info.get('torch','')))+'; CUDA '+escape(str(info.get('cuda','')))+'. Exactly one GPU was visible to training. Raw logs, hardware.txt, per-game config.json and run_provenance.json are retained.','SmallText')
    else:
        p('The retained local CPU results are exploratory. Final deepdish results must carry hardware and per-game device records.','SmallText')
    page();title('1. Algorithm and experimental design')
    heading('Bellman learning and architecture')
    p('For a replay tuple (s, a, r, s_next, terminated), the target is y = r + gamma (1 - terminated) max_b Q_target(s_next, b). A Huber loss fits Q_online(s,a) to y; gradients are clipped to norm 10. The default is ordinary DQN. The optional --double switch chooses the next action with the online network and evaluates it with the target network; it is not enabled for the reported default run.')
    table([['Setting','CartPole-v1','ALE/MsPacman-v5'],['State / network','4 state values; 128-128 ReLU MLP','4 x 88 x 80 pixels; conv 32/64/64; dense 512'],['Convolutions','Not applicable','8x8/stride4, 4x4/stride2, 3x3/stride1'],['Gamma; Adam learning rate','0.95; 0.001','0.99; 0.0001'],['Replay capacity; warmup','50,000; 1,000 transitions','100,000; 20,000 transitions'],['Batch; optimization frequency','64; every 4 environment steps','64; every 4 environment steps'],['Target hard sync','Every 500 environment steps','Every 10,000 environment steps'],['Epsilon schedule','1.0 to 0.02 over 20,000 steps','1.0 to 0.01 over 1,000,000 steps'],['Reward used for learning','Raw +1 survival reward','Sign-clipped reward'],['Performance metric','Raw undiscounted score','Raw undiscounted game score']],[125,177,177])
    heading('Image processing and version fidelity')
    p('The assignment crop and grayscale sum are preserved, including removal of pixels whose RGB sum is 448. The transformed image is stored as uint8; subtracting 128 and dividing by 128 in the CNN gives the same centered intensities as the supplied int8 transform, then scales them. Four consecutive preprocessed observations provide motion information. Reset repeats the first frame four times. The ALE/v5 environment already repeats an action for four frames and uses sticky-action probability 0.25; no second frame-skip wrapper is added [1].')
    heading('Evaluation protocol')
    p('Training seed is 42. Every 25 CartPole episodes, a greedy 20-episode validation uses seeds starting at 200000. MsPacman validation uses 10 episodes every 100 training episodes. The best validation checkpoint is frozen before 500 greedy test episodes, seeds 100000-100499. A uniform-random baseline uses those same seeds. Standard deviations are sample SD (ddof=1). Evaluation does not optimize parameters or fill replay.')
    page();title('2. CartPole training evidence')
    d=root/'cartpole'
    if (d/'train.csv').exists():
        rows=read_csv(d/'train.csv'); rewards=np.array([float(r['reward']) for r in rows]); cfg=json.loads((d/'config.json').read_text())
        p(f'The retained run contains {len(rows)} episodes and {int(rows[-1]["steps"]):,} environment steps, on {escape(cfg["device_used"])}. Mean reward in the first 100 episodes: {rewards[:100].mean():.2f}; last 100: {rewards[-100:].mean():.2f}. Training includes epsilon exploration, so its scores need not match the greedy evaluation scores.')
        image(d/'q_values.png');image(d/'rewards.png')
        p('Figure 1. Maximum action values are summarized over visited states and an independent fixed 128-observation probe. The fixed probe separates value changes from changes in visited-state distribution. Figure 2. Raw episode reward with the trailing mean of exactly 100 episodes; the mean is omitted until 100 observations exist.','SmallText')
        p('For constant reward 1 and gamma=.95, infinite-horizon value saturates at 1/(1-.95)=20. A high or flat Q curve therefore cannot establish 500-step control. Non-monotone training and validation can reflect bootstrapping instability; held-out survival and robustness provide stronger behavioral evidence.','SmallText')
    else:p('CartPole training has not been run.')
    page();title('3. CartPole held-out evaluation')
    if (d/'evaluation.json').exists():
        e=json.loads((d/'evaluation.json').read_text()); baseline=json.loads((d/'random.json').read_text()) if (d/'random.json').exists() else None
        table([['Policy','Episodes','Mean raw reward','Sample SD','500-step fraction'],['Greedy DQN',e['n'],f'{e["mean"]:.3f}',f'{e["std"]:.3f}',f'{100*e["success_500"]:.1f}%']]+([['Uniform random',baseline['n'],f'{baseline["mean"]:.3f}',f'{baseline["std"]:.3f}',f'{100*baseline["success_500"]:.1f}%']] if baseline else []),[105,65,105,95,109])
        if baseline:
            er=read_csv(d/'evaluation.csv'); br=read_csv(d/'random.csv'); delta=np.array([float(x['reward'])-float(y['reward']) for x,y in zip(er,br)])
            p(f'Matched-seed mean improvement over random: {delta.mean():.3f} steps (approximate 95% interval {delta.mean()-1.96*delta.std(ddof=1)/np.sqrt(len(delta)):.3f} to {delta.mean()+1.96*delta.std(ddof=1)/np.sqrt(len(delta)):.3f}). This interval is conditional on one trained model, not uncertainty across training seeds.')
        if e['success_500']==1:p('All 500 held-out greedy episodes reached the 500-step limit in this test. This meets the requested survival behavior on the sampled reset distribution; it is not a guarantee for all states or future seeds.')
        else:p('Some held-out episodes did not reach 500 steps. The strict goal of maintaining the pole upright for every evaluated episode is not demonstrated; the failure rate is retained rather than hidden.')
        image(d/'histogram.png',450);image(d/'calibration.png',450)
        er=read_csv(d/'evaluation.csv'); q0=np.array([float(x['q_start']) for x in er]); mc=np.array([float(x['discounted_return']) for x in er])
        extra=''
        if e['success_500']==1:
            extra=' For 500-step trajectories the omitted constant-reward tail is only .95^500/.05, about 1.46e-10; a larger calibration gap cannot be explained by that tail.'
        p(f'Mean initial Q={q0.mean():.4f}; mean realized discounted return={mc.mean():.4f}; mean error={(q0-mc).mean():.4f}.'+extra, 'SmallText')
        p('Figure 3. Held-out raw-return histogram. Figure 4 (non-standard). Initial greedy Q versus realized discounted return, colored by episode length, and its error versus length. Because a time limit cuts off an otherwise bootstrapped process, a truncated return is a censored prefix; a positive gap alone does not prove optimistic bias. Near Q=20, value calibration may look accurate even when survival varies substantially.','SmallText')
    else:p('The required 500-episode held-out evaluation has not been completed.')
    page();title('4. Edge cases and validation')
    v=json.loads((root/'validation.json').read_text()) if (root/'validation.json').exists() else {}
    heading('Experiments with known predictions')
    p('A deterministic two-state MDP isolates temporal credit assignment. Action a0 moves s0 to s1 for reward zero and s1 to a terminal state for reward c; a1 terminates for zero at either state. The exact values are Q(s0,a0)=gamma*c, Q(s1,a0)=c, and zero for both quit actions. The same optimizer update as the game agent learns this MDP for 1,500 updates, with target synchronization every 25 updates. The pass criterion is maximum absolute Q error below 0.02.')
    if v:
        table([['Gamma','Reward c','Max absolute Q error','Expected effect']]+[[c['gamma'],c['reward_scale'],f'{c["max_absolute_error"]:.3g}','No future value' if c['gamma']==0 else ('Zero value everywhere' if c['reward_scale']==0 else ('Triple value scale' if c['reward_scale']==3 else 'Correct delayed reward'))] for c in v['analytic_learning']],[65,70,140,204])
    p('Changing gamma from .95 to .99 changes the delayed value by exactly .04 when c=1. Setting gamma=0 removes delayed reward. Tripling c triples values; zero rewards must collapse them to zero. These counterfactuals test the learning mechanism beyond data formatting. They do not prove that a CNN can solve MsPacman.')
    heading('Termination, truncation, histories and stability')
    p('Physical pole failure and Atari game-over stop bootstrapping. Time-limit truncation ends a rollout but retains the successor bootstrap [2]. With r=1, gamma=.95 and successor value 10, the measured targets are 1.0 for termination and 10.5 for truncation. The final successor is saved before reset. Losing an Atari life is not treated as game-over; this run learns full-game returns.')
    p('Empty/undersized replay is handled by warmup. The target network is detached and synchronized at explicit step intervals. In the pixel replay, serial and episode identifiers prevent histories crossing resets or overwritten ring entries. A numbered-frame intervention gives exact expected histories after two resets and ring wrap; all checked histories agree. The first frame is repeated when an episode has fewer than four frames. A contrast pixel verifies RGB-sum 448 is removed. Huber loss and gradient clipping limit the influence of outlying TD errors; raw evaluation rewards are never clipped.')
    heading('First-ten-step trace and how to reproduce')
    p('Run python validate.py --checkpoint results/cartpole/best.pt --out results/validation.json. Each evaluation command saves its first ten transitions in *_first10.json, including action values, reward, action, termination and truncation. Compare changes in state with the chosen action, and inspect whether +1 survival rewards propagate toward the value scale 20. Image traces store pixel summaries. README.md describes every experiment, its predicted effect, tolerance and output.')
    page();title('5. Robustness and MsPacman status')
    if 'cartpole_angle_perturbation' in v:
        p('A controlled state intervention adds a pole-angle offset immediately after reset, keeping every other initial state component and each of 100 evaluation seeds fixed. This challenges the narrow training reset distribution without changing the physics or reward.')
        table([['Angle offset (rad)','Mean return','Sample SD','500-step success']]+[[r['angle_offset_rad'],f'{r["mean"]:.2f}',f'{r["std"]:.2f}',f'{r["success_500"]*100:.1f}%'] for r in v['cartpole_angle_perturbation']],[120,120,120,119])
        image(d/'robustness.png',400)
        robust=v['cartpole_angle_perturbation']
        if all(r['success_500']==1 for r in robust):
            interpretation='All tested offsets achieved 500 steps in every sampled episode, so observed intervals collapse to zero width.'
        else:
            interpretation='The table retains failures and the plot shows how robustness changes with initial angle.'
        p('Figure 5. '+interpretation+' These tests cover the stated offsets only and do not estimate variability across training seeds.','SmallText')
    heading('MsPacman results and limitations')
    md=root/'mspacman'
    if not (md/'evaluation.json').exists():
        p('NOT COMPLETED. The image network accepts four 88x80 frames and emits nine action values; the local two-episode CPU smoke run verifies the environment/network path only. It has no trained-policy score, learning trend or convergence claim. The required maximum-Q plot, reward/100-episode-average plot and histogram of 500 post-training episodes remain pending. No CartPole figure is reused as MsPacman evidence.')
        sp=root/'pixel_smoke.json'
        if sp.exists():
            px=json.loads(sp.read_text());p(f'A separate real-image integration experiment collected {px["steps"]} ALE steps, then fitted one fixed 64-transition batch for {px["optimizer_updates"]} updates with a frozen target. Mean Huber loss fell from {px["initial_5_mean_huber_loss"]:.5f} (first five updates) to {px["final_5_mean_huber_loss"]:.5f} (last five), and target parameters stayed unchanged. This verifies the pixel training path, not game learning.', 'SmallText')
        p('The deepdish script is configured for 6,000 full-game episodes. Inspect whether raw scores and held-out validation improve, compare the selected model with a matched 500-episode random baseline, and extend or tune only on validation evidence if needed. Reward clipping bounds targets but may trade off collecting many small pellets against high-value events; report that limitation alongside raw scores.')
    else:
        m=json.loads((md/'evaluation.json').read_text()); p(f'Greedy evaluation: {m["n"]} episodes, mean raw score {m["mean"]:.3f}, sample SD {m["std"]:.3f}. Full curves and calibration follow on the next pages.')
        training=read_csv(md/'train.csv'); candidates=[x for x in training if x['validation_mean']]
        chosen=max(candidates,key=lambda x:float(x['validation_mean']))
        p(f'MsPacman training completed {len(training):,} episodes and {int(training[-1]["steps"]):,} decision steps. The selected checkpoint is from episode {chosen["episode"]}, with mean validation score {float(chosen["validation_mean"]):.2f} over ten separate validation seeds. Selection occurred before the 500-episode final test.', 'SmallText')
    if (md/'evaluation.json').exists():
        page();title('6. MsPacman learning curves'); image(md/'q_values.png'); image(md/'rewards.png');p('Figure 6. Maximum action values in clipped-reward units. Figure 7. Raw game scores with the trailing mean of 100 episodes.','SmallText')
        page();title('7. MsPacman held-out performance');image(md/'histogram.png',450);image(md/'calibration.png',450);p('Figure 8. Histogram of 500 greedy raw scores. Figure 9. Non-standard initial-value calibration in sign-clipped discounted-return units; game-over and time-limit observations are distinguished in raw CSV.','SmallText')
        if (md/'random.json').exists():
            r=json.loads((md/'random.json').read_text());p(f'Random baseline: mean {r["mean"]:.3f}, sample SD {r["std"]:.3f}, {r["n"]} matched-seed episodes. Greedy minus random mean score: {m["mean"]-r["mean"]:.3f}.')
        ee=read_csv(md/'evaluation.csv'); rr=read_csv(md/'random.csv')
        delta=np.array([float(x['reward'])-float(y['reward']) for x,y in zip(ee,rr)]); se=delta.std(ddof=1)/np.sqrt(len(delta))
        qstart=np.array([float(x['q_start']) for x in ee]); realized=np.array([float(x['discounted_return']) for x in ee])
        p(f'The matched-seed mean score improvement is {delta.mean():.2f}, with approximate 95% interval [{delta.mean()-1.96*se:.2f}, {delta.mean()+1.96*se:.2f}]. This measures episode variability conditional on this one trained model.', 'SmallText')
        p(f'Mean initial Q is {qstart.mean():.3f}, versus mean realized clipped discounted return {realized.mean():.3f}; mean calibration error is {(qstart-realized).mean():.3f}. The initial image is identical across these resets, so Q is constant while later stochastic trajectories vary. The observed underprediction shows that strong raw scores and increasing training Q values do not establish value calibration or full convergence.', 'SmallText')
        rows=read_csv(md/'train.csv'); rew=np.array([float(x['reward']) for x in rows]);p(f'Training first-100 mean {rew[:100].mean():.2f}; last-100 mean {rew[-100:].mean():.2f}. A higher final mean is evidence of improvement in this run; one run cannot establish stability across training seeds.')
    if 'mspacman_pixel_interventions' in v:
        page();title('8. MsPacman visual-information interventions')
        pix=v['mspacman_pixel_interventions']; cond=pix['conditions']
        p('The selected frozen CNN is evaluated on 50 paired, separate seeds (500000-500049) under three inference conditions: intact image history; the latest frame repeated four times, removing temporal differences while retaining current appearance; and a uniform intensity-128 input, removing all state information. The true environment and raw score remain unchanged. These seeds are not used to select or retune the policy.')
        table([['Condition','Mean raw score','Sample SD','Action entropy (bits)']]+[[name,f'{r["mean"]:.2f}',f'{r["std"]:.2f}',f'{max(0.,r["action_entropy_bits"]):.3f}'] for name,r in cond.items()],[150,105,105,119])
        for name in ['repeat_latest_frame','constant_image']:
            diff=cond[name]['intact_minus_condition'];lo,hi=diff['approx_95_interval']
            p(f'Intact minus {name}: paired mean score difference {diff["mean"]:.2f}, approximate 95% interval [{lo:.2f}, {hi:.2f}].', 'SmallText')
        if all(cond[name]['intact_minus_condition']['approx_95_interval'][0]>0 for name in ['repeat_latest_frame','constant_image']):
            p('Both paired intervals exclude zero in favor of intact history. This supports dependence on image information and temporal differences in this frozen model. It is stronger evidence than a rising training reward alone, while remaining subject to the distribution-shift limitation below.')
        p('Prediction: if useful image and temporal information affect control, intact inputs should outperform the corresponding removals. Positive paired differences support that hypothesis in this trained model; a near-zero or negative difference limits the claim. The constant-image policy is deterministic and receives identical network inputs every step, so it should choose one fixed action (entropy zero), an additional check on evaluation semantics. These are distribution-shift interventions, not equal-budget retraining experiments; they do not prove optimal behavior or isolate every visual feature.')
        p('Reproduce: python validate.py --checkpoint results/cartpole/best.pt --pixel-checkpoint results/mspacman/best.pt --pixel-episodes 50 --out results/validation.json. Per-episode diagnostic scores are retained in results/mspacman/ablation.csv, and validation.json records the exact checkpoint SHA-256. The required 500-episode intact-policy test remains a separate experiment.','SmallText')
    page();title('Additional three months: validation plan')
    heading('Month 1: reproducibility and causality')
    p('Train at least five independent seeds per environment with a fixed budget in environment frames and GPU hours. Reserve new final test seeds before tuning. Compare DQN, Double DQN, no target network and no replay under equal budgets; log raw scores, TD errors, target drift and confidence intervals. Test whether removing replay or the target network worsens instability rather than assuming it does. Use a zero-reward control and a tiny deterministic image MDP to verify the CNN/replay path learns known pixel-based values, extending the present analytic control.')
    heading('Month 2: state coverage and reward design')
    p('For CartPole, sweep angle, angular velocity, cart position and cart velocity on a predeclared grid, reporting return and failure probability per region. Vary mass and force within documented ranges to measure robustness without claiming these modified environments are the graded task. For MsPacman, compare one versus four frames, sign clipping versus scaled raw rewards, and replay capacities under equal frame budgets. Evaluate ghost/fruit encounters, life loss, repeated-action behavior and stagnation, using actual image trajectories and reward-event logs rather than only final scores.')
    heading('Month 3: calibration, stress tests and statistics')
    p('At fixed archived states, collect repeated greedy rollouts with independent environment randomness to estimate return distributions and compare them with learned Q. Separate terminal and censored trajectories. Use paired bootstrap intervals for score improvements and aggregate over independent training seeds. Predeclare stopping rules and minimum meaningful improvements, then evaluate once on a final untouched seed set. Stress interrupted-job recovery, GPU determinism, memory limits, replay overwrite and long games; implement exact replay/optimizer/RNG resume checkpoints before relying on them for long training.')
    heading('Submission checklist')
    p('The final artifact must include the actual public repository identity, every source file, raw evidence, both sets of required figures, 500-episode statistics for each trained network, this validation discussion and verified deepdish single-GPU provenance. The report generator refuses --final when required local evidence is missing, but the author must separately verify public repository access and upload the PDF to Canvas.')
    heading('References')
    for text in ['[1] Farama Foundation. MsPacman, Arcade Learning Environment. https://ale.farama.org/environments/ms_pacman/', '[2] Farama Foundation. Handling Time Limits. https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/', '[3] Mnih et al. Human-level control through deep reinforcement learning. Nature 518, 529-533 (2015). https://www.nature.com/articles/nature14236']:
        p(escape(text),'SmallText')
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#b9c8ce'));canvas.line(54,43,558,43);canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#52636d'));canvas.drawString(54,30,'Diego HW2 | Assignment 2'+(' | INCOMPLETE DRAFT' if draft else ''));canvas.drawRightString(558,30,str(doc.page))
    SimpleDocTemplate(str(output),pagesize=letter,rightMargin=54,leftMargin=54,topMargin=42,bottomMargin=56,title='Diego - Assignment 2 - Deep Q-Network',author='').build(story,onFirstPage=footer,onLaterPages=footer)
    print(output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--github-user',default='');p.add_argument('--repo-url',default='');p.add_argument('--final',action='store_true');build(p.parse_args())
