import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F 
import os
import time
import timeit
import copy
import numpy as np 
from torch.nn import ModuleList
from torch.nn import Conv2d
from torch.nn import LeakyReLU
from net.UDLSSPI005_step1 import LSSPI_one
import torch.nn as nn
import torch
import random

class LSSPI_two(nn.Module):
    def __init__(self, path):
        super(LSSPI_two, self).__init__()
        self.LSSPI_var=LSSPI_one(picSize=256,patchSize=8)
        self.LSSPI_U = LSSPI_one(picSize=256,patchSize=8)
        # if args.pre_train_step1 != '.':
        self.LSSPI_var.load_state_dict(torch.load(path), strict=True)
        self.LSSPI_U.load_state_dict(torch.load(path), strict=True)
    def forward(self, x):
        with torch.no_grad():
            var = self.LSSPI_var(x)
        x = self.LSSPI_U(x)
        x = x[0]
        return [x, var[1]]


    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

