#!/usr/bin/env python3
from functools import partial

import gym
import ptan
import numpy as np
import argparse
import collections
# from tensorboardX import SummaryWriter
import torch.nn as nn
import torch
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import time
import sys

from dataclasses import dataclass
from fastai.data.all import *
from fastrl.data import *

from fastrl.basic_agents import *
from fastrl.actorcritic.a3c_data import *
from fastrl.async_data import *

class RewardTracker:
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):pass
        # self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        # print(reward)
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))
        sys.stdout.flush()
        # if epsilon is not None:
        #     self.writer.add_scalar("epsilon", epsilon, frame)
        # self.writer.add_scalar("speed", speed, frame)
        # self.writer.add_scalar("reward_100", mean_reward, frame)
        # self.writer.add_scalar("reward", reward, frame)
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False

# from lib import common
class LinearA2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(LinearA2C, self).__init__()

        self.policy = nn.Sequential(
            nn.Linear(input_shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(input_shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        fx=x.float()
        return self.policy(fx),self.value(fx)


GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128

REWARD_STEPS = 4
CLIP_GRAD = 0.1

PROCESSES_COUNT = 3 # 4
NUM_ENVS = 15 # 15

# if True:
#     ENV_NAME = "PongNoFrameskip-v4"
#     NAME = 'pong'
#     REWARD_BOUND = 18
# else:
#     ENV_NAME = "BreakoutNoFrameskip-v4"
#     NAME = "breakout"
#     REWARD_BOUND = 400
#
if True:
    ENV_NAME = "CartPole-v1"
    NAME = 'cartpole'
    REWARD_BOUND = 200
else:
    ENV_NAME = "CartPole-v1"
    NAME = "cartpole2"
    REWARD_BOUND = 200



# def make_env():
#     return ptan.common.wrappers.wrap_dqn(gym.make(ENV_NAME))

def make_env():
    _env=gym.make(ENV_NAME)
    # _env.reset()
    return _env # common.PixelObservationWrapper(_env,boxify=True)


TotalReward = collections.namedtuple('TotalReward', field_names='reward')


class Debug(ActorCriticAgent):
    def __call__(self, *args, **kwargs):
        print( *args, **kwargs)
        return super().__call__( *args, **kwargs)


def _data_func(net, device, train_queue):
    envs = [make_env() for _ in range(NUM_ENVS)]
    # agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], device=device, apply_softmax=True)
    agent=ActorCriticAgent(net,device='cuda')
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    for exp in exp_source:
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            # print(exp.last_state is None,len(new_rewards),flush=True)
            train_queue.put(TotalReward(reward=np.mean(new_rewards)))
        train_queue.put(exp)

def data_func(net,device,train_queue):
    agent=ActorCriticAgent(model=net,device='cuda')
    experience_block = partial(FirstLastExperienceBlock, a=0, seed=0, n_steps=4, dls_kwargs={'bs': 1, 'num_workers': 0,
                                                                                             'verbose': False,
                                                                                             'indexed': True,
                                                                                             'shuffle_train': False})
    # print(experience_block,agent)
    blk=IterableDataBlock(blocks=(experience_block(agent=agent)),
                          splitter=FuncSplitter(lambda x:False))
    dls=blk.dataloaders(['CartPole-v1']*NUM_ENVS,device='cpu',n=128*100)
    while True:
        for xb in dls[0]:
            # print(xb)
            xb=[o.cpu().numpy()[0] for o in xb]
            xb=[ExperienceFirstLast(state=xb[0],action=xb[1],reward=xb[2],last_state=xb[3] if not xb[4] else None,done=xb[4],episode_reward=xb[5])]

            new_rewards = [o.episode_reward for o in xb if o.done and int(o.episode_reward)!=0]
            if new_rewards:
                # print(exp.last_state is None,len(new_rewards),flush=True)
                train_queue.put(TotalReward(reward=np.mean(new_rewards)))

            for x in xb:
                # print(x)
                train_queue.put(x)


ExperienceFirstLastNew = collections.namedtuple('ExperienceFirstLastNew', ('s', 'a', 'r', 'sp','d'))

def unbatch(batch, net, last_val_gamma, device='cpu'):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []

    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            # if exp.last_state is None:print(exp)
            last_states.append(np.array(exp.last_state, copy=False))
        # else:
        #     print(exp,'is done, so skipping')
    states_v = torch.FloatTensor(states).to(device)
    actions_t = torch.LongTensor(actions).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        # print(last_states)
        last_states_v = torch.FloatTensor(last_states).to(device)
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        # print(last_vals_v.data.cpu().numpy().mean())
        rewards_np[not_done_idx] += last_val_gamma * last_vals_np

    # print(len(not_done_idx),len(rewards_np),len(last_states))
    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_t, ref_vals_v

class MultiProcessTfm(Transform):
    def __init__(self,
                 n: int = 1,
                 n_processes: int = 1, process_cls=None,
                 cancel=None,
                 verbose: str = False,
                 regular_get: bool = False,
                 tracker=None
                 ):
        store_attr(but='process_cls')
        # self.process_cls=process_cls
        # self.n_processes=n_processes
        self.process_cls=ifnone(process_cls,DataFitProcess)
        self.queue = mp.JoinableQueue(maxsize=self.n_processes)
        # self.cancel = ifnone(self.cancel,mp.Event())
        # self.pipe_in, self.pipe_out = mp.Pipe(False) if self.verbose else (None, None)
        # self.cached_items = []
        # self._place_holder_out = None
        # self.step_idx=0
        # mp.set_start_method('spawn', force=True)

    def setup(self, items: TfmdSource, train_setup=False):
        print('setting up')
        # self.cancel.clear()
        # if len(items.items) != 0 and not issubclass(items.items[0].__class__, DataFitProcess):
        #     self.cached_items = deepcopy(items.items)
        self.reset(items)

    def reset(self, items: TfmdSource, train_setup=False):
        self.step_idx = 0
        # self.close(items)
        # self.cancel.clear()
        self.queue = mp.JoinableQueue(maxsize=self.n_processes)
        items.items = [self.process_cls(start=True, train_queue=self.queue)
                       for _ in range(self.n_processes)]
        # if not all([p.is_alive() for p in items.items]): raise CancelFitException()

    def close(self, items: TfmdSource):
        self.step_idx = 0
        # print('close')
        # self.cancel.set()
        # for o in [p for p in items.items if issubclass(p.__class__, DataFitProcess)]:
        #     o.termijoin()
        #     del o
        #     torch.cuda.empty_cache()
        # try:
        #     while not self.queue.empty(): self.queue.get()
        # except (ConnectionResetError, FileNotFoundError, EOFError, ConnectionRefusedError, RuntimeError):
        #     pass
        # items.items.clear()

    def encodes(self, o):
        pv('encodes {o}', self.verbose)
        while True:
            if not self.cancel.is_set():
                o = self.queue.get()
                self._place_holder_out = ifnone(self._place_holder_out, o)
                if isinstance(o, TotalReward):
                    if tracker.reward(o.reward, self.step_idx):sys.exit()
                    continue
                return o
            else:
                raise CancelFitException()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=False,default='cartpole', help="Name of the run")
    args = parser.parse_args()
    device = "cuda" # if args.cuda else "cpu"

    # writer = SummaryWriter(comment="-a3c-data_" + NAME + "_" + args.name)

    env = make_env()
    net = LinearA2C(env.observation_space.shape, env.action_space.n).to(device)
    net.share_memory()

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    # train_queue = mp.JoinableQueue(maxsize=PROCESSES_COUNT)
    # data_proc_list = []
    # process_cls = partial(DataFitProcess, net=net, device=device,data_fit=data_func)
    # data_proc_list=[process_cls(start=True,train_queue=train_queue) for _ in range(PROCESSES_COUNT)]

    # for _ in range(PROCESSES_COUNT):
    #     data_proc = DataFitProcess(True,data_func,net=net, device=device, train_queue=train_queue)
    #     # data_proc.start()
    #     data_proc_list.append(data_proc)

    batch = []
    step_idx = 0
    batch_num = 0
    try:
        with RewardTracker(None, stop_reward=REWARD_BOUND) as tracker:
            @dataclass
            class FakeSource(object):
                def __init__(self):
                    self.items: str = ['CartPole-v0'] * NUM_ENVS


            tfm = MultiProcessTfm(n_processes=PROCESSES_COUNT,
                                  tracker=tracker,
                                  process_cls=partial(DataFitProcess, net=net,device=device,
                                                      data_fit=data_func))
            tfm.setup(FakeSource())
            # process_cls = partial(DataFitProcess, net=net, device=device,data_fit=data_func)
            # data_proc_list=[tfm.process_cls(start=True,train_queue=tfm.queue) for _ in range(PROCESSES_COUNT)]

            # with ptan.common.utils.TBMeanTracker(None, batch_size=100) as tb_tracker:
            while True:
                # train_entry = train_queue.get()
                train_entry= tfm.queue.get()
                # print(train_entry)
                if isinstance(train_entry, TotalReward):
                    if tracker.reward(train_entry.reward, step_idx):
                        break
                    continue

                step_idx += 1
                # print(train_entry)
                batch.append(train_entry)
                if len(batch) < BATCH_SIZE:
                    continue
                batch_num+=1
                # states_v1, actions_t, vals_ref_v1 = \
                #     common.unpack_batch(batch, net, last_val_gamma=GAMMA**REWARD_STEPS, device=device)
                # print(batch)
                states_v, actions_t, vals_ref_v = \
                    unbatch(batch, net, last_val_gamma=GAMMA**REWARD_STEPS, device=device)
                # print(np.array([_b.reward for _b in batch]).mean(), vals_ref_v.float().mean())
                # print(vals_ref_v.mean(), np.mean([o.reward for o in batch]))
                # print(states_v.shape,actions_t.shape,vals_ref_v.shape)
                # vals_ref_v=vals_ref_v.squeeze(1)
                # print(states_v.float().mean(),actions_t.float().mean(),vals_ref_v.float().mean())
                batch.clear()
                print(batch_num)

                optimizer.zero_grad()
                logits_v, value_v = net(states_v)
                # print(logits_v.mean(),logits_v.shape,value_v.mean(),value_v.shape)

                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                log_prob_v = F.log_softmax(logits_v, dim=1)
                adv_v = vals_ref_v - value_v.detach()
                log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions_t]
                loss_policy_v = -log_prob_actions_v.mean()
                prob_v = F.softmax(logits_v, dim=1)
                entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()
                loss_v = entropy_loss_v + loss_value_v + loss_policy_v

                # print(entropy_loss_v,loss_policy_v,loss_value_v)
                loss_v.backward()
                # getBack(loss_v.grad_fn)
                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                # print(step_idx)
                optimizer.step()
                # if batch_num>1:
                #     getModelconf(net,True)
                #     if batch_num>2:raise Exception
                # print(batch_num,loss_v.detach(),entropy_loss_v.detach(),loss_value_v.detach(),loss_policy_v.detach())

                    # tb_tracker.track("advantage", adv_v, step_idx)
                    # tb_tracker.track("values", value_v, step_idx)
                    # tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                    # tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                    # tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                    # tb_tracker.track("loss_value", loss_value_v, step_idx)
                    # tb_tracker.track("loss_total", loss_v, step_idx)
    finally:
        tfm.close(FakeSource())
        # for p in data_proc_list:
        #     p.terminate()
        #     p.join()
