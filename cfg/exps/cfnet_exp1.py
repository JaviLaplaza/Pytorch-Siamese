#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 13:14:49 2018

@author: jlaplaza
"""


exp_name = 'cfnet_exp1'

pretrained_fname = 'darknet19.weights.npz'

start_step = 0
lr_decay_epochs = {0, 60, 90}
lr_decay = 1./10

max_epoch = 1

weight_decay = 0.0005
momentum = 0.9
init_learning_rate = 1e-3


# dataset
batch_size = 2