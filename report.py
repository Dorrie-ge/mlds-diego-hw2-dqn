"""Regenerate my first-person report from retained evidence, in Times New Roman."""
import argparse,csv,json,hashlib
from pathlib import Path
from functools import partial
from xml.sax.saxutils import escape
import numpy as np
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen.canvas import Canvas
from report_fonts import register_report_fonts

BASE=Path(__file__).resolve().parent


def read_csv(path):
    with open(path) as f:return list(csv.DictReader(f))


def build(a):
    root=BASE/'results'; run=BASE/'artifacts'/'run'; missing=[]
    if not a.github_user or not a.repo_url:missing.append('GitHub username and public repository URL are required.')
    if not (run/'hardware.txt').exists():missing.append('GPU hardware record is missing.')
    for game in ['cartpole','mspacman']:
        d=root/game
        for name in ['train.csv','config.json','evaluation.csv','evaluation.json','random.csv','random.json','q_values.png','rewards.png','histogram.png','calibration.png']:
            if not (d/name).exists():missing.append(f'{game}: missing {name}')
        if (d/'config.json').exists():
            cfg=json.loads((d/'config.json').read_text())
            if cfg['environment']!=('CartPole-v1' if game=='cartpole' else 'ALE/MsPacman-v5'):missing.append(f'{game}: incorrect environment version')
        for policy in ['evaluation','random']:
            if (d/f'{policy}.json').exists() and (d/f'{policy}.csv').exists():
                stats=json.loads((d/f'{policy}.json').read_text()); raw=read_csv(d/f'{policy}.csv')
                if len(raw)!=500 or stats['n']!=500:missing.append(f'{game}: {policy} requires 500 episodes')
                elif not np.isclose(np.mean([float(x['reward']) for x in raw]),stats['mean']):missing.append(f'{game}: inconsistent {policy} statistics')
    vp=root/'validation.json';v=json.loads(vp.read_text()) if vp.exists() else {}
    if not v.get('analytic_learning') or not v.get('cartpole_angle_perturbation'):missing.append('Behavioral validation is missing.')
    if not missing:
        ckpt=root/'mspacman'/'best.pt';old=v.get('mspacman_pixel_interventions',{})
        expected=hashlib.sha256(ckpt.read_bytes()).hexdigest() if ckpt.exists() else old.get('checkpoint_sha256')
        if not old or old.get('checkpoint_sha256')!=expected:
            if not ckpt.exists():missing.append('Pixel interventions require the selected checkpoint or retained evidence.')
            else:
                from validate import pixel_ablation
                v['mspacman_pixel_interventions']=pixel_ablation(ckpt,root/'mspacman'/'ablation.csv',50)
                vp.write_text(json.dumps(v,indent=2))
        train=read_csv(root/'mspacman'/'train.csv')
        if len(train)<4000:missing.append('MsPacman training is shorter than the stated experiment.')
        ev=json.loads((root/'mspacman'/'evaluation.json').read_text());rnd=json.loads((root/'mspacman'/'random.json').read_text())
        if ev['mean']<=rnd['mean']:missing.append('MsPacman does not improve over random.')
    if missing:raise SystemExit('Report refused:\n'+'\n'.join(missing))
    register_report_fonts()
    styles=getSampleStyleSheet()
    for style in styles.byName.values():style.fontName='TNR'
    styles.add(ParagraphStyle(name='Text',fontName='TNR',fontSize=10.5,leading=13.7,spaceAfter=8))
    styles.add(ParagraphStyle(name='SmallText',fontName='TNR',fontSize=9,leading=11.3,spaceAfter=6))
    styles['Heading1'].fontName='TNR-Bold';styles['Heading1'].fontSize=17;styles['Heading1'].leading=20
    styles['Heading2'].fontName='TNR-Bold';styles['Heading2'].fontSize=12;styles['Heading2'].leading=15
    styles['Heading1'].textColor=colors.HexColor('#16384b')
    story=[]
    def p(text,style='Text'):story.append(Paragraph(text,styles[style]))
    def title(text):p(text,'Heading1')
    def heading(text):p(text,'Heading2')
    def page():story.append(PageBreak())
    def table(rows,widths):
        cells=[[Paragraph(escape(str(x)),styles['SmallText']) for x in row] for row in rows]
        for j,x in enumerate(rows[0]):cells[0][j]=Paragraph('<b>'+escape(str(x))+'</b>',styles['SmallText'])
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'TNR'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f0f3')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#93a8b2')),('BOTTOMPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6)]))
        story.extend([t,Spacer(1,8)])
    def image(path,width=480):
        from PIL import Image as PILImage
        with PILImage.open(path) as im:w,h=im.size
        story.extend([Image(str(path),width=width,height=width*h/w),Spacer(1,5)])
    cp=root/'cartpole';ms=root/'mspacman'
    ce=json.loads((cp/'evaluation.json').read_text());cr=json.loads((cp/'random.json').read_text())
    me=json.loads((ms/'evaluation.json').read_text());mr=json.loads((ms/'random.json').read_text())
    title('Assignment 2 - Deep Q-Network')
    p('Diego HW2 | MLDS | Due November 11, 2026, 11:59 pm','Heading2')
    p('GitHub username: '+escape(a.github_user))
    p('Repository: '+escape(a.repo_url.rstrip('/').split('/')[-1]))
    p('Public source URL: '+escape(a.repo_url))
    heading('My objective and results')
    p('I implemented DQN with a neural action-value function, experience replay and a separately synchronized target network for both required environments. I used gamma = 0.95 for CartPole-v1 and gamma = 0.99 for image-based ALE/MsPacman-v5 with nine actions. I base every result in this report on my retained experiment logs.')
    table([['My greedy policy','Test episodes','Mean raw reward','Sample SD'],['CartPole-v1',ce['n'],f'{ce["mean"]:.2f}',f'{ce["std"]:.2f}'],['ALE/MsPacman-v5',me['n'],f'{me["mean"]:.2f}',f'{me["std"]:.2f}']],[160,90,120,109])
    heading('How I reproduce and audit my work')
    p('I provide my source code, raw CSV/JSON, figures and reproduction commands in the public repository. I reproduce the deepdish environment with setup_deepdish.sh, select one GPU with CUDA_VISIBLE_DEVICES, and run bash run_deepdish.sh with GITHUB_USER and REPO_URL set. I describe individual commands and the Times New Roman font prerequisite in docs/REPRODUCING.md. I select checkpoints on validation seeds before inspecting the 500 final test episodes.')
    info=json.loads((run/'run_provenance.json').read_text())
    p(f'I ran training on {escape(info["hostname"])} with one {escape(info["gpu_name"])} visible to PyTorch {escape(info["torch"])} and CUDA {escape(info["cuda"])}. I retain the execution log, hardware information and timestamps in artifacts/run/; I keep each game\'s exact configuration beside its result CSVs.','SmallText')
    page();title('1. My DQN implementation')
    heading('My learning rule and architectures')
    p('I sample replay tuples (s, a, r, s_next, terminated) and form y = r + gamma (1 - terminated) max_b Q_target(s_next, b). I minimize Huber loss between Q_online(s,a) and y, clip gradient norms to 10, and synchronize a detached target network at fixed intervals. I report ordinary DQN; I include an optional --double implementation for separate future ablations but did not enable it in these runs.')
    table([['Setting','My CartPole model','My MsPacman model'],['Input / network','4 values; MLP 128-128 ReLU','4 x 88 x 80 pixels; CNN; dense 512'],['Convolutions','Not applicable','32/64/64 channels; 8x8/4, 4x4/2, 3x3/1'],['Gamma; Adam learning rate','0.95; 0.001','0.99; 0.0001'],['Replay capacity; warmup','50,000; 1,000 transitions','100,000; 20,000 transitions'],['Batch; update frequency','64; every 4 decision steps','64; every 4 decision steps'],['Target synchronization','Every 500 decision steps','Every 10,000 decision steps'],['Epsilon schedule','1 to .02 over 20,000 steps','1 to .01 over 1,000,000 steps'],['Training rewards','Raw survival rewards','Sign-clipped game rewards'],['Reported performance','Raw undiscounted rewards','Raw undiscounted game scores']],[125,177,177])
    heading('How I process images and preserve the task')
    p('I use the assignment\'s crop and grayscale transform, including removal of pixels with RGB sum 448. I store uint8 intensities and center/scale them in the CNN as (pixel - 128)/128. I stack four consecutive observations for motion information and repeat the initial frame at reset. I retain v5\'s default four-frame action repeat and sticky-action probability .25 [1]; I do not add a second action-repeat wrapper. I apply gamma per agent decision step.')
    heading('How I separate training, selection and testing')
    p('I train with seed 42. I validate CartPole on 20 greedy episodes every 25 training episodes and MsPacman on ten every 100, using seeds starting at 200000. I freeze the best validation checkpoint before 500 greedy tests on seeds 100000-100499. I evaluate uniform-random policies on the same test seeds, without optimization or replay updates. I report sample standard deviations with ddof = 1 and keep all test seeds separate from checkpoint selection.')
    page();title('2. My CartPole training results')
    rows=read_csv(cp/'train.csv');rewards=np.array([float(r['reward']) for r in rows])
    p(f'I trained CartPole for {len(rows):,} episodes and {int(rows[-1]["steps"]):,} decision steps on CUDA. I observed mean rewards of {rewards[:100].mean():.2f} in my first 100 episodes and {rewards[-100:].mean():.2f} in my last 100. I retain exploration in the training curve, so I assess the selected greedy policy separately.')
    image(cp/'q_values.png');image(cp/'rewards.png')
    p('Figures 1-2. I plot maximum action values over visited states and a fixed 128-observation probe, followed by raw rewards and the exact trailing 100-episode mean. I omit the moving average until 100 episodes exist. I use the fixed probe to separate value changes from changes in the visited-state distribution.','SmallText')
    p('I expect constant-reward CartPole values to approach 1/(1-.95) = 20. I therefore do not treat a flat or high Q curve as proof of 500-step control. I use held-out survival and state interventions to check behavior despite non-monotone bootstrapped training.','SmallText')
    page();title('3. My CartPole held-out evaluation')
    table([['Policy','Episodes','Mean reward','Sample SD','500-step fraction'],['My greedy DQN',ce['n'],f'{ce["mean"]:.3f}',f'{ce["std"]:.3f}',f'{100*ce["success_500"]:.1f}%'],['Uniform random',cr['n'],f'{cr["mean"]:.3f}',f'{cr["std"]:.3f}',f'{100*cr["success_500"]:.1f}%']],[110,65,100,95,109])
    er=read_csv(cp/'evaluation.csv');br=read_csv(cp/'random.csv');delta=np.array([float(x['reward'])-float(y['reward']) for x,y in zip(er,br)]);se=delta.std(ddof=1)/np.sqrt(len(delta))
    p(f'I measured a paired improvement of {delta.mean():.3f} steps over random (approximate 95% interval {delta.mean()-1.96*se:.3f} to {delta.mean()+1.96*se:.3f}). I interpret this interval as episode variability for one trained model, rather than uncertainty across training seeds. I reached 500 steps in every held-out episode; I do not claim this guarantees success for every possible state.')
    image(cp/'histogram.png',440);image(cp/'calibration.png',440)
    q=np.array([float(x['q_start']) for x in er]);mc=np.array([float(x['discounted_return']) for x in er])
    p(f'Figures 3-4. I show the reward histogram and my non-standard value-calibration diagnostic. I compare initial greedy Q with realized discounted return and plot the error against episode length. I measured mean initial Q {q.mean():.4f}, mean realized return {mc.mean():.4f}, and mean error {(q-mc).mean():.4f}. For these 500-step trajectories, I bound the omitted constant-reward tail by .95^500/.05, about 1.46e-10; that tail cannot explain my larger calibration gap.','SmallText')
    page();title('4. How I validate learning and edge cases')
    heading('My experiments with exact predictions')
    p('I isolate temporal credit assignment with a deterministic two-state MDP. I let a0 move s0 to s1 for zero reward and then terminate from s1 with reward c; I let a1 quit for zero reward from either state. I expect Q(s0,a0) = gamma*c, Q(s1,a0) = c, and zero quit values. I train through the same update function for 1,500 updates, synchronize the target every 25 updates, and require maximum absolute error below .02.')
    table([['Gamma','Reward c','My maximum Q error','Prediction I test']]+[[c['gamma'],c['reward_scale'],f'{c["max_absolute_error"]:.3g}','No delayed value' if c['gamma']==0 else ('Zero values' if c['reward_scale']==0 else ('Triple value scale' if c['reward_scale']==3 else 'Delayed reward'))] for c in v['analytic_learning']],[65,70,140,204])
    p('I test gamma .95 and .99, remove delayed rewards with gamma 0, triple the reward scale, and set every reward to zero. I recover the predicted values in all cases. I use these counterfactuals to verify discounting and reward propagation beyond format checks; I do not treat the toy linear model as proof of image-based control.')
    heading('How I handle terminal states, time limits and replay')
    p('I stop bootstrapping at physical pole failure or Atari game-over, but retain it at time-limit truncation [2]. I measured targets 1.0 and 10.5 for a controlled terminal/truncated pair with reward 1, gamma .95 and successor value 10. I save the final successor before reset and treat life loss as part of a full Atari game rather than game-over.')
    p('I wait through replay warmup before updating. I detach my target network, synchronize it explicitly, and use Huber loss and gradient clipping for outlying TD errors. I reconstruct pixel histories with serial and episode identifiers, pad short histories with the first frame, and exclude overwritten histories. I verified exact numbered-frame histories across resets and ring wrap, and checked that the contrast transform removes RGB sum 448. I leave evaluation scores unclipped.')
    trace=json.loads((cp/'evaluation_first10.json').read_text());actions=', '.join(str(t['action']) for t in trace)
    p(f'I also retain the first ten test transitions. In my first CartPole trace I observed actions [{actions}], total reward {sum(t["reward"] for t in trace):.0f}, and no early termination. I provide states, Q values, rewards and termination flags in the trace JSON so I can inspect the interaction step by step. I reproduce the analytic and behavioral checks with validate.py; I document each experiment in docs/REPRODUCING.md.','SmallText')
    page();title('5. My robustness checks and MsPacman run')
    robust=v['cartpole_angle_perturbation']
    p('I add a controlled pole-angle offset immediately after reset while keeping the other state components and each of 100 held-out seeds fixed. I use this intervention to test recovery outside the narrow training reset distribution without changing the physics or reward.')
    table([['My angle offset (rad)','Mean return','Sample SD','500-step success']]+[[r['angle_offset_rad'],f'{r["mean"]:.2f}',f'{r["std"]:.2f}',f'{100*r["success_500"]:.1f}%'] for r in robust],[130,110,110,129])
    image(cp/'robustness.png',400)
    p('Figure 5. I reached 500 steps in every sampled episode at all five offsets, so my observed intervals collapse to zero width. I interpret this as evidence within the tested range, not as coverage of all state perturbations or variability across training seeds.','SmallText')
    heading('My completed MsPacman experiment')
    tr=read_csv(ms/'train.csv');selected=max([x for x in tr if x['validation_mean']],key=lambda x:float(x['validation_mean']))
    p(f'I completed {len(tr):,} MsPacman episodes and {int(tr[-1]["steps"]):,} decision steps. I selected episode {selected["episode"]} using its ten-episode validation mean of {float(selected["validation_mean"]):.2f}, before testing. I then measured a 500-episode mean raw score of {me["mean"]:.2f} and sample SD {me["std"]:.2f}.')
    page();title('6. My MsPacman learning curves')
    image(ms/'q_values.png');image(ms/'rewards.png')
    rew=np.array([float(x['reward']) for x in tr])
    p(f'Figures 6-7. I plot maximum action values in clipped-reward units and raw game scores with the trailing 100-episode mean. I increased my mean raw training score from {rew[:100].mean():.2f} in the first 100 episodes to {rew[-100:].mean():.2f} in the last 100. I interpret this trend as learning in this run; I do not infer full convergence or stability across training seeds from one curve.','SmallText')
    page();title('7. My MsPacman held-out performance')
    image(ms/'histogram.png',440);image(ms/'calibration.png',440)
    ee=read_csv(ms/'evaluation.csv');rr=read_csv(ms/'random.csv');delta=np.array([float(x['reward'])-float(y['reward']) for x,y in zip(ee,rr)]);se=delta.std(ddof=1)/np.sqrt(len(delta))
    qs=np.array([float(x['q_start']) for x in ee]);realized=np.array([float(x['discounted_return']) for x in ee])
    p(f'Figures 8-9. I show my 500-episode raw-score histogram and my non-standard calibration diagnostic in clipped discounted-return units. I measured random mean {mr["mean"]:.2f} and sample SD {mr["std"]:.2f}; my paired score improvement is {delta.mean():.2f}, with approximate 95% interval [{delta.mean()-1.96*se:.2f}, {delta.mean()+1.96*se:.2f}]. I condition this interval on the one trained model.','SmallText')
    p(f'I measured mean initial Q {qs.mean():.3f}, versus realized clipped discounted return {realized.mean():.3f}, for mean error {(qs-realized).mean():.3f}. I obtain the same initial image across resets, so my initial Q is constant while later trajectories vary. I observe substantial underprediction despite improved raw scores, which limits my claim about value calibration and convergence. I distinguish terminal and time-limit trajectories in the raw CSV because finite observed prefixes can censor bootstrapped returns.','SmallText')
    page();title('8. How I test dependence on image information')
    pix=v['mspacman_pixel_interventions'];cond=pix['conditions'];labels={'intact':'Intact history','repeat_latest_frame':'Latest frame repeated','constant_image':'Constant image'}
    p('I freeze my selected CNN and evaluate 50 paired diagnostic seeds (500000-500049) with intact history, the latest frame repeated four times, and a uniform intensity-128 input. I remove temporal differences in the second condition and all state information in the third while leaving the true environment unchanged. I do not use these seeds to select or retune my policy.')
    table([['My inference condition','Mean score','Sample SD','Action entropy (bits)']]+[[labels[name],f'{r["mean"]:.2f}',f'{r["std"]:.2f}',f'{max(0.,r["action_entropy_bits"]):.3f}'] for name,r in cond.items()],[155,100,100,124])
    for name in ['repeat_latest_frame','constant_image']:
        diff=cond[name]['intact_minus_condition'];lo,hi=diff['approx_95_interval']
        p(f'I measured intact minus {labels[name].lower()}: paired mean difference {diff["mean"]:.2f}, approximate 95% interval [{lo:.2f}, {hi:.2f}].','SmallText')
    p('I predicted that removing useful appearance or motion information would reduce scores. I found both paired intervals above zero in favor of intact history, supporting dependence on image information and temporal differences in my frozen model. I also recovered zero action entropy with the constant image, as I expect from a deterministic policy receiving identical inputs at every step.')
    p('I treat these interventions as distribution shifts, not equal-budget retraining ablations. I therefore do not claim that they prove optimal control or isolate every visual feature. I retain the raw diagnostic scores in results/mspacman/ablation.csv and the exact checkpoint SHA-256 in validation.json. I reproduce the checks with python validate.py --checkpoint results/cartpole/best.pt --pixel-checkpoint results/mspacman/best.pt --pixel-episodes 50 --out results/validation.json. I keep the required 500-episode test separate from these 50-seed diagnostics.','SmallText')
    page();title('What I would do with three additional months')
    heading('Month 1: I would test reproducibility and causality')
    p('I would train at least five independent seeds per environment under fixed budgets in environment frames and GPU hours, reserving new final seeds before tuning. I would compare DQN, Double DQN, no target network and no replay with equal budgets, recording scores, TD errors, target drift and intervals. I would test whether removing stabilization actually worsens behavior. I would extend my analytic control with a tiny deterministic image MDP and a zero-reward control to verify the CNN/replay path against known values.')
    heading('Month 2: I would test coverage and reward design')
    p('I would sweep CartPole angle, angular velocity, cart position and cart velocity on a predefined grid, reporting return and failure probability by region. I would vary mass and force within documented ranges as robustness tests while retaining the original graded task for final results. For MsPacman, I would retrain one-frame and four-frame models, compare sign clipping with scaled raw rewards, and vary replay capacity under equal frame budgets. I would inspect ghost/fruit encounters, life loss, action repetition and stagnation using trajectories and event logs.')
    heading('Month 3: I would test calibration, stress and statistics')
    p('I would archive fixed states and collect repeated greedy rollouts with independent environment randomness to estimate return distributions against Q. I would separate terminal and censored trajectories, bootstrap paired score improvements, and aggregate across training seeds. I would predeclare stopping rules and minimum meaningful gains, then evaluate once on untouched final seeds. I would stress interrupted jobs, GPU determinism, memory limits, replay overwrite and long games, implementing exact replay/optimizer/RNG checkpoints before relying on resumed training.')
    heading('My submission and reproducibility record')
    p('I include my public repository identity, all source, raw evidence, required figures for both environments, 500-episode statistics, validation discussion and deepdish single-GPU provenance. I provide this PDF for Canvas submission and keep the complete report preview in README.md. I preserve the experimental limitations rather than treating successful policy evaluation as proof of perfect Q calibration.')
    heading('References')
    for text in ['[1] Farama Foundation. MsPacman, Arcade Learning Environment. https://ale.farama.org/environments/ms_pacman/','[2] Farama Foundation. Handling Time Limits. https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/','[3] Mnih et al. Human-level control through deep reinforcement learning. Nature 518, 529-533 (2015). https://www.nature.com/articles/nature14236']:p(escape(text),'SmallText')
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#b9c8ce'));canvas.line(54,43,558,43);canvas.setFont('TNR',8);canvas.setFillColor(colors.HexColor('#52636d'));canvas.drawString(54,30,'MLDS | Diego HW2 | '+a.github_user);canvas.drawRightString(558,30,str(doc.page))
    output=BASE/'report.pdf'
    SimpleDocTemplate(str(output),pagesize=letter,rightMargin=54,leftMargin=54,topMargin=42,bottomMargin=56,title='Assignment 2 - Deep Q-Network',author=a.github_user,initialFontName='TNR').build(story,onFirstPage=footer,onLaterPages=footer,canvasmaker=partial(Canvas,initialFontName='TNR'))
    print(output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--github-user',required=True);p.add_argument('--repo-url',required=True);p.add_argument('--final',action='store_true');build(p.parse_args())
