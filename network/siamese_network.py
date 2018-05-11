#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 16:52:51 2018

@author: jlaplaza
"""

import sys

sys.path.append("../")

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
#from utils.conv2d_local import Conv2dLocal

from utils.timer import Timer



class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2),
            nn.BatchNorm2d(96, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.BatchNorm2d(256, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(256, 192, kernel_size=3, stride=1),
            nn.BatchNorm2d(192, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=1),
            nn.BatchNorm2d(192, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128, momentum=0.1)
            )
        
        self.loss = None
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        

    def forward(self, input1, input2, gt = None):
        output1 = self.cnn(input1)
        #print("output1.shape = " + str(output1.shape))

        output2 = self.cnn(input2)
        #print("output2.shape = " + str(output2.shape))
        
        # define probabilty map dimensions
        b_size = output1.size()[0]
        oH = output1.size()[2] - output2.size()[2] + 1
        oW = output1.size()[3] - output2.size()[3] + 1
        output = Variable(torch.zeros((b_size, 1, oH, oW)))
        
        """
        output = F.conv2d(output1, output2)
        """
        
        # perform cross-correlation operation
        for i in range(oH):
            for j in range(oW):
                output[:, :, i, j] = torch.sum(torch.mul(output1[:, :, i:i+6, j:j+6], output2)) / output2.nelement()
                
                        
        
                
        if self.training:
            self.loss = nn.SoftMarginLoss(size_average=True)(output, gt)
            #print(type(self.loss))
        
        return output
        
    

if __name__ == '__main__':
    
    t = Timer()
    
    batch_size = 3
    model = Siamese()
    # net.load_from_npz('models/yolo-voc.weights.npz')
    # model.load_from_npz(cfg.pretrained_model, num_conv=18)
    model.train()
    print(model.loss)
    a = Variable(torch.randn((batch_size, 3, 255, 255)))
    b = Variable(torch.randn((batch_size, 3, 127, 127)))
    
    gt = torch.ones([batch_size, 1, 17, 17])
    #print(type(gt))
    gt = gt*-1
    gt[:, :, 8, 8] = 1
    gt = Variable(gt)
    
    t_for = t.tic()
    c = model(a, b, gt)
    t_for = t.toc()
    print("Forward time: %f s" % t_for)
    print(c.shape) #must be (batch_size, 1, 17, 17)
    print(model.loss.item())
    del model, a, b, gt, c
    
