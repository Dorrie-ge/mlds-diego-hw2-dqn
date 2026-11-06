"""Scientific validation: analytic Bellman solutions and controlled interventions."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from dqn import seed_all,bellman,update,Replay,preprocess,make_env,load_model,policy


def analytic():
    seed_all(17)
    # Deterministic two-state MDP: a0 moves to rewarding terminal; a1 quits for zero.
    # s0,a0 -> s1,r0; s1,a0 -> terminal,r=scale. Both a1 transitions terminate.
    states=torch.eye(2)
    s=states[[0,0,1,1]]; ns=states[[1,0,1,1]]
    a=torch.tensor([0,1,0,1]); terminal=torch.tensor([0.,1.,1.,1.])
    cases=[]
    for gamma,scale in [(.95,1.),(.99,1.),(0.,1.),(.95,3.),(.95,0.)]:
        net=nn.Linear(2,2); target=nn.Linear(2,2); target.load_state_dict(net.state_dict()); target.requires_grad_(False)
        opt=torch.optim.Adam(net.parameters(),lr=.02)
        r=torch.tensor([0.,0.,scale,0.]); batch=[s,a,r,ns,terminal]
        for i in range(1500):
            update(net,target,opt,batch,gamma)
            if (i+1)%25==0: target.load_state_dict(net.state_dict())
        actual=net(states).detach().numpy(); expected=np.array([[gamma*scale,0],[scale,0]])
        error=float(abs(actual-expected).max())
        cases.append(dict(gamma=gamma,reward_scale=scale,expected=expected.tolist(),learned=actual.tolist(),max_absolute_error=error,pass_tolerance_002=error<.02))
    # Quantitative counterfactual: a time-limit observation with V=10 must bootstrap.
    mask=bellman(torch.tensor([1.,1.]),torch.tensor([1.,0.]),torch.tensor([10.,10.]),.95).tolist()
    # Pixel history test with two short episodes and ring wrap, using identifiable frames.
    replay=Replay(8,(4,2,2),True)
    for i in range(14):
        replay.add(np.full((4,2,2),i,np.uint8),0,0,np.full((4,2,2),i+1,np.uint8),False,i//3)
    histories={str(i):replay.stack(i)[:,0,0].tolist() for i in range(9,14)}
    expected_hist={str(i):[max(i//3*3,i-k) for k in [3,2,1,0]] for i in range(9,14)}
    pixels=np.zeros((210,160,3),np.uint8); pixels[1,0]=[210,164,74]
    contrast=int(preprocess(pixels)[0,0])
    result=dict(analytic_learning=cases,terminal_vs_truncation_targets=dict(actual=mask,expected=[1.,10.5]),replay_history=dict(actual=histories,expected=expected_hist,passed=histories==expected_hist),background_pixel_after_contrast=contrast)
    assert all(c['pass_tolerance_002'] for c in cases),result
    assert np.allclose(mask,[1.,10.5]) and histories==expected_hist and contrast==0,result
    return result


def perturb(checkpoint,n=100):
    net,game=load_model(checkpoint,torch.device('cpu'))
    if game!='cartpole': raise ValueError('Initial-state perturbation is CartPole-only')
    env=make_env(game); rows=[]
    # Same seeds, controlled angle offset before the first action. Independent of tuning seeds.
    for angle in [-.15,-.075,0.,.075,.15]:
        rewards=[]
        for ep in range(n):
            s,_=env.reset(seed=400000+ep)
            env.unwrapped.state=np.asarray(env.unwrapped.state,dtype=float).copy()
            env.unwrapped.state[2]+=angle; s=np.asarray(env.unwrapped.state,dtype=np.float32)
            total=0
            for t in range(500):
                s,r,term,trunc,_=env.step(policy(net,s,'cpu')); total+=r
                if term or trunc: break
            rewards.append(total)
        rows.append(dict(angle_offset_rad=angle,n=n,mean=float(np.mean(rewards)),std=float(np.std(rewards,ddof=1)),success_500=float(np.mean(np.array(rewards)==500)),rewards=rewards))
    env.close(); return rows

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default='results/validation.json'); p.add_argument('--checkpoint'); a=p.parse_args()
    r=analytic()
    if a.checkpoint: r['cartpole_angle_perturbation']=perturb(a.checkpoint)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
