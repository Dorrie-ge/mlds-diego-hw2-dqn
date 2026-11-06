"""Real-ALE fixed-target fitting check; not an Atari learning benchmark."""
import json,random
from pathlib import Path
import numpy as np
import torch
from dqn import seed_all,make_env,QNet,Replay,update
seed_all(73)
env=make_env('mspacman'); s,_=env.reset(seed=73)
net=QNet('mspacman',9); target=QNet('mspacman',9); target.load_state_dict(net.state_dict()); target.requires_grad_(False)
replay=Replay(1024,s.shape,True); ep=0; raw_score=0
for step in range(1024):
    a=random.randrange(9); ns,r,term,trunc,_=env.step(a)
    replay.add(s,a,float(np.sign(r)),ns,term,ep); s=ns; raw_score+=r
    if term or trunc: ep+=1; s,_=env.reset(seed=73+ep)
env.close(); batch=replay.sample(64,'cpu'); initial={k:v.clone() for k,v in target.state_dict().items()}
opt=torch.optim.Adam(net.parameters(),lr=.0001); losses=[]
for _ in range(60): losses.append(update(net,target,opt,batch,.99))
assert all(torch.equal(v,initial[k]) for k,v in target.state_dict().items())
assert np.isfinite(losses).all() and np.mean(losses[-5:])<np.mean(losses[:5])
r=dict(environment='ALE/MsPacman-v5',steps=1024,episodes_completed=ep,raw_reward_during_collection=raw_score,fixed_batch_size=64,optimizer_updates=60,initial_5_mean_huber_loss=float(np.mean(losses[:5])),final_5_mean_huber_loss=float(np.mean(losses[-5:])),target_unchanged=True,limitation='Fixed-target minibatch fitting verifies CNN/replay/update integration; it does not establish game learning or convergence.')
p=Path(__file__).resolve().parent/'results'/'pixel_smoke.json'; p.write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
