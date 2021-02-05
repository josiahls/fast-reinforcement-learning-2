# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/20a_qlearning.dqn.ipynb (unless otherwise specified).

__all__ = ['LinearDQN', 'ExperienceReplay', 'EpsilonTracker', 'calc_target', 'DQNTrainer', 'DQNLearner']

# Cell
import torch.nn.utils as nn_utils
from fastai.torch_basics import *
from fastai.data.all import *
from fastai.basics import *
from dataclasses import field,asdict
from typing import List,Any,Dict,Callable
from collections import deque
import gym
import torch.multiprocessing as mp
from copy import deepcopy
from torch.optim import *

from ..data import *
from ..async_data import *
from ..basic_agents import *
from ..learner import *
from ..metrics import *
from ..ptan_extension import *

if IN_NOTEBOOK:
    from IPython import display
    import PIL.Image

# Cell
class LinearDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(LinearDQN, self).__init__()

        self.policy = nn.Sequential(
            nn.Linear(input_shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self,x):
        fx=x.float()
        return self.policy(fx)

class ExperienceReplay(Callback):
    def __init__(self,sz=100,bs=128,starting_els=1,max_steps=1):
        store_attr()
        self.queue=deque(maxlen=int(sz))
        self.max_steps=max_steps

    def before_fit(self):
        self.learn.agent.warming_up=True
        while len(self.queue)<self.starting_els:
            for i,o in enumerate(self.dls.train):
                batch=[ExperienceFirstLast(state=o[0][i],action=o[1][i],reward=o[2][i],
                                    last_state=o[3][i], done=o[4][i],episode_reward=o[5][i],steps=o[6][i])
                                    for i in range(len(o[0]))]
#                 print(self.max_steps,max([o.steps for o in batch]))
                for _b in batch: self.queue.append(_b)
                if len(self.queue)>self.starting_els:break
        self.learn.agent.warming_up=False

#     def after_epoch(self):
#         print(len(self.queue))
    def before_batch(self):
#         print(len(self.queue))
        b=list(self.learn.xb)+list(self.learn.yb)
        batch=[ExperienceFirstLast(state=b[0][i],action=b[1][i],reward=b[2][i],
                                last_state=b[3][i], done=b[4][i],episode_reward=b[5][i],
                                steps=b[6][i])
                                for i in range(len(b[0]))]
        for _b in batch: self.queue.append(_b)
        idxs=np.random.randint(0,len(self.queue), self.bs)
        self.learn.sample_yb=[deepcopy(self.queue[i]) for i in idxs]

# Cell
class EpsilonTracker(Callback):
    def __init__(self,e_stop=0.2,e_start=1.0,e_steps=5000,current_step=0):
        store_attr()

    def before_fit(self):
        self.learn.agent.a_selector.epsilon=self.e_start

    def after_step(self):
        self.learn.agent.a_selector.epsilon=max(self.e_stop,self.e_start-self.current_step/self.e_steps)
        self.current_step+=1

# Cell
def calc_target(net, local_reward,next_state,done,discount):
    if done: return local_reward
    next_q_v = net(next_state.float().unsqueeze(0))
    best_q = next_q_v.max(dim=1)[0].item()
    return local_reward + discount * best_q

class DQNTrainer(Callback):
    def __init__(self,target_fn=None):
        self.target_fn=ifnone(target_fn,calc_target)

    def after_pred(self):
        s,a,r,sp,d,er,steps=(self.learn.xb+self.learn.yb)
        exps=[ExperienceFirstLast(*o) for o in zip(*(self.learn.xb+self.learn.yb))]
        batch_targets=[self.target_fn(self.learn.model, exp.reward, exp.last_state,exp.done,self.learn.discount)
                         for exp in exps]

        s_v = s.float()
        q_v = self.learn.model(s_v)
        t_q=q_v.data.numpy().copy()
        t_q[range(len(exps)), a] = batch_targets
        target_q_v = torch.tensor(t_q)
        self.learn._yb=self.learn.yb
        self.learn.yb=(target_q_v,)
        self.learn.pred=q_v
#         print(*self.learn.yb,self.learn.pred)
#         print(self.learn.pred,self.learn.yb)
#         print(self.learn._yb,self.learn.yb[0])

    def after_loss(self):self.learn.yb=self.learn._yb

# Cell
class DQNLearner(AgentLearner):
    def __init__(self,dls,discount=0.99,**kwargs):
        store_attr()
        self.target_q_v=[]
        super().__init__(dls,loss_func=nn.MSELoss(),**kwargs)