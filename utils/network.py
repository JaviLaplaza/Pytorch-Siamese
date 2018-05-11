#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 12:48:32 2018

@author: jlaplaza
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

import cfg.config as cfg



class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, same_padding=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding=padding)
        self.relu = nn.LeakyReLU(0.1, inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Conv2d_BatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, same_padding=False):
        super(Conv2d_BatchNorm, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01)
        self.relu = nn.LeakyReLU(0.1, inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
    

def np_to_variable(x, use_cuda=cfg.use_cuda, dtype=torch.FloatTensor, volatile=False):
    v = Variable(torch.from_numpy(x).type(dtype), volatile=volatile)
    if use_cuda:
        v = v.cuda()
    return v

def save_net(fname, model):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in list(model.state_dict().items()):
        h5f.create_dataset(k, data=v.cpu().numpy())