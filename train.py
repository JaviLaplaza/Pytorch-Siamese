#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 13:02:20 2018

@author: jlaplaza
"""
import os
import datetime

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataset.cfnet_dataset import CFNetDataset
from network.siamese_network import Siamese

import cfg.config as cfg
from utils.timer import Timer





# dataset object creation
dataset_root_dir = cfg.DATASET_DIR
dataset = CFNetDataset(dataset_root_dir)
print('Dataset loaded successfully.')

# defining how many batches per epoch do we hace according to dataset lenght and batch size 
batch_per_epoch = len(dataset)/cfg.batch_size


# creating dataloader object
dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, drop_last=True)


# creating model
model = Siamese()

if cfg.use_cuda:
    model.cuda()

# setting model to train mode
model.train()

print("Setting model to Training Mode.")


# defining training parameters
step_cnt = 0
start_epoch = 0
lr = cfg.init_learning_rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.momentum,
                            weight_decay=cfg.weight_decay)
cnt = 0
t = Timer()

# ground truth will be the same for all batches, since search images are centred
gt = torch.ones([cfg.batch_size, 1, 17, 17])
if cfg.use_cuda:
    gt = gt.cuda()

gt = gt*-1
gt[:, :, 8, 8] = 1
gt = Variable(gt)



# iterate through steps
for epoch in range(start_epoch, cfg.max_epoch):
    
    # iterating through mini-batches
    for i_batch, sample_batched in enumerate(dataloader):
        
        # Start timer
        t.tic()
        
        # preprocess data
        sample_batched['z'] = Variable(sample_batched['z'].type(torch.FloatTensor)).permute(0, 3, 1, 2)
        sample_batched['x'] = Variable(sample_batched['x'].type(torch.FloatTensor)).permute(0, 3, 1, 2)
        
        # forward 
        model(sample_batched['x'],
              sample_batched['z'], 
              gt)
        
        # calculate loss
        loss = model.loss
        
            
        # zero grad variable
        optimizer.zero_grad()
        
        # calculate grads
        loss.backward()
        
        # update weights
        optimizer.step()
        
        
        cnt += 1
        step_cnt += 1
        duration = t.toc()
        
        if epoch % cfg.disp_interval == 0:
            #loss /= cnt
            print(('epoch %d[%d/%d], loss: %.3f (%.2f s/batch, rest:%s)' %
                   (epoch, step_cnt, batch_per_epoch, loss.item(), duration, 
                    str(datetime.timedelta(seconds=int((batch_per_epoch - step_cnt) * duration))))))
            
            
            cnt = 0
            t.clear()
            
            if epoch > 0 and (epoch % batch_per_epoch == 0):
                if epoch in cfg.lr_decay_epochs:
                    lr *= cfg.lr_decay
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                                momentum=cfg.momentum,
                                                weight_decay=cfg.weight_decay)
                    
                    PATH = os.path.join(cfg.train_output_dir, "cfnet_%s_%i" % (cfg.exp_name, epoch))
                    
                    torch.save(model.state_dict(), PATH)
                    
                    print(('save model: %s' % PATH))
                    step_cnt = 0
                    
                    

del model