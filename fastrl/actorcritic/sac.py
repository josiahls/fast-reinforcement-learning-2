# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/14_actorcritic.sac.ipynb (unless otherwise specified).

__all__ = ['weights_init_', 'ValueNetwork', 'QNetwork', 'GaussianPolicy', 'DeterministicPolicy', 'create_log_gaussian',
           'logsumexp', 'soft_update', 'hard_update', 'SAC', 'ExperienceReplay', 'SACCriticTrainer', 'SACLearner',
           'LOG_SIG_MAX', 'LOG_SIG_MIN', 'epsilon']

# Cell
import torch.nn.utils as nn_utils
from fastai.torch_basics import *
import torch.nn.functional as F
from fastai.data.all import *
from fastai.basics import *
from dataclasses import field,asdict
from typing import List,Any,Dict,Callable
from collections import deque
import gym
import torch.multiprocessing as mp
from torch.optim import *
from dataclasses import dataclass

from ..data import *
from ..async_data import *
from ..basic_agents import *
from ..learner import *
from ..metrics import *
from fastai.callback.progress import *
from ..ptan_extension import *

from torch.distributions import *

if IN_NOTEBOOK:
    from IPython import display
    import PIL.Image

# Cell
# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state.float()))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


# Cell
class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state.float(), action.float()], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2



# Cell
class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state.float()))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


# Cell
class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state.float()))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)



# Cell
def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


# Cell
from pprint import pprint

class SAC(BaseAgent):
    def __init__(self, num_inputs, action_space, gamma,tau,alpha,policy='gaussian',
                automatic_entropy_tuning=True,target_update_interval=1,hidden_size=100,lr=0.0003,max_steps=1):

        self.action_space=action_space
        self.warming_up=False
        self.max_steps=max_steps

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.policy_type = policy
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.device=default_device()

        self.critic = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        self.updates=0

    def select_action(self, state, evaluate=False):
        if self.warming_up: return self.action_space.sample()
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def __call__(self,s,asl):
        return self.select_action(s),asl

    def update_parameters(self, *yb, learn):
        # Sample a batch from memory
#         state_batch, action_batch, reward_batch, next_state_batch, mask_batch = learn.memory.sample(batch_size=batch_size)
        batch=learn.sample_yb
#         pprint(batch)
        state_batch=torch.stack([o.state.to(device=default_device()) for o in batch]).float()
        next_state_batch=torch.stack([o.last_state.to(device=default_device()) for o in batch]).float()
        action_batch=torch.stack([o.action.to(device=default_device()) for o in batch]).float()
        reward_batch=torch.stack([o.reward.to(device=default_device()) for o in batch]).float().unsqueeze(1)
        mask_batch=torch.stack([o.done.to(device=default_device()) for o in batch]).float().unsqueeze(1)

#         print(state_batch.shape,next_state_batch.shape,action_batch.shape,reward_batch.shape,mask_batch.shape)
#         state_batch = torch.FloatTensor(state_batch).to(self.device)
#         next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
#         action_batch = torch.FloatTensor(action_batch).to(self.device)
#         reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
#         mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1-mask_batch) * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if self.updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        self.updates+=1
#         print(self.updates)

        return qf1_loss+ qf2_loss+ policy_loss+ alpha_loss+ alpha_tlogs

# Cell
class ExperienceReplay(Callback):
    def __init__(self,sz=100,bs=128,starting_els=1):
        store_attr()
        self.queue=deque(maxlen=int(sz))

    def before_fit(self):
        self.learn.agent.warming_up=True
        while len(self.queue)<self.starting_els:
            for i,o in enumerate(self.dls.train):
                batch=[ExperienceFirstLast(state=o[0][i],action=o[1][i],reward=o[2][i],
                                    last_state=o[3][i], done=(o[4][i] and self.learn.agent.max_steps!=o[6][i]),episode_reward=o[5][i],steps=o[6][i])
                                    for i in range(len(o[0]))]
                print(self.learn.agent.max_steps,max([o.steps for o in batch]))
                for _b in batch: self.queue.append(_b)
                if len(self.queue)>self.starting_els:break
        self.learn.agent.warming_up=False

    def after_epoch(self):
        print(len(self.queue))
    def before_batch(self):
#         print(len(self.queue))
        b=list(self.learn.xb)+list(self.learn.yb)
        batch=[ExperienceFirstLast(state=b[0][i],action=b[1][i],reward=b[2][i],
                                last_state=b[3][i], done=(b[4][i] and self.learn.agent.max_steps!=b[6][i]),episode_reward=b[5][i],
                                steps=b[6][i])
                                for i in range(len(b[0]))]
        for _b in batch: self.queue.append(_b)
        idxs=np.random.randint(0,len(self.queue), self.bs)
        self.learn.sample_yb=[self.queue[i] for i in idxs]

# Cell
class SACCriticTrainer(Callback):
    def after_batch(self):
        self.learn.dls.bs=1
        for d in self.learn.dls.loaders: d.bs=1

    def after_loss(self):raise CancelBatchException

# Cell
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class SACLearner(AgentLearner):
    def __init__(self,dls,agent=None,reward_scale=2,**kwargs):
        store_attr()
        super().__init__(dls,loss_func=partial(self.agent.update_parameters,learn=self),model=agent.policy,**kwargs)