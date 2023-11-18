import numpy as np

from pathlib import Path
from model import Model
from tokenizer import CharTokenizer
from utils import get_formated_date, load_stat_dict
from torch.optim import Optimizer
from data import DataLoader
from typing import Callable, Union
from torch.nn import Module
from functools import wraps
from hprams import hprams
from tqdm import tqdm
import torch
import os
from search import GreedyDecode
from warprnnt_pytorch import RNNTLoss

inputs = torch.from_numpy(np.load('inputs.npy')).to('cuda')
inputs_len = torch.from_numpy(np.load('inputs_len.npy')).to('cuda')
targets = torch.from_numpy(np.load('targets.npy')).to('cuda')
targets_len = torch.from_numpy(np.load('targets_len.npy')).to('cuda')

a = [1, 11, 10, 23,  5, 12,  2, 20, 23, 25, 30,  2, 13, 21,  2, 13, 25, 28,
          2, 10,  7,  2, 24, 14,  7, 23, 20, 13,  5,  2, 26, 14,  2, 23, 20, 14,
          7, 23,  5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0]

print(len(a))