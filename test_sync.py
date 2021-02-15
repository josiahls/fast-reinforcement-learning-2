# Python native modules
import os
# Third party libs
from fastcore.all import *
from fastai.torch_basics import *
from fastai.data.all import *
from fastai.basics import *
from torch.utils.data import Dataset
from torch import nn
# Local modules
from fastrl.data.block import *


import numpy as np
import gym
import torch.multiprocessing as mp

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except Exception: pass
    ds=TestDataset(DQN(),device=default_device())
    dl=DataLoader(ds,num_workers=1)
    for x in dl:
        print(x)
    print(ds.pids)
