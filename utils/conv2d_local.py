#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 12:52:18 2018

@author: jlaplaza
"""



import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn


class CorrelationLayer():

    def __init__(self, args=None, padding=0, kernel_size = 1, max_displacement=20, stride_1=1, stride_2=1):
        # TODO generilize kernel size (right now just 1)
        # TODO generilize stride_1 (right now just 1), cause there is no downsample layer in pytorch
        #super(CorrelationLayer,self).__init__(args)
        self.pad = padding
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride_1 = stride_1
        self.stride_2 = stride_2


    def forward(self, x_1, x_2):
        """
        Arguments
        ---------
        x_1 : 4D torch.Tensor (bathch channel height width)
        x_2 : 4D torch.Tensor (bathch channel height width)
        """
        x_1 = x_1.transpose(1,2).transpose(2,3)
        x_2 = F.pad(x_2, tuple([self.pad for _ in range(4)])).transpose(1,2).transpose(2,3)
        mean_x_1 = torch.mean(x_1,3) 
        mean_x_2 = torch.mean(x_2,3) 
        sub_x_1 = x_1.sub(mean_x_1.expand_as(x_1))
        sub_x_2 = x_2.sub(mean_x_2.expand_as(x_2))
        st_dev_x_1 = torch.std(x_1,3) 
        st_dev_x_2 = torch.std(x_2,3)
        
        # TODO need optimize
        out_vb = torch.zeros(1)
        _y=0
        _x=0
        while _y < self.max_displacement*2+1:
            while _x < self.max_displacement*2+1:
                c_out = (torch.sum(sub_x_1*sub_x_2[:,_x:_x+x_1.size(1),
                    _y:_y+x_1.size(2),:],3) / 
                (st_dev_x_1*st_dev_x_2[:,_x:_x+x_1.size(1),
                    _y:_y+x_1.size(2),:])).transpose(2,3).transpose(1,2)
                out_vb = torch.cat((out_vb,c_out),1) if len(out_vb.size())!=1 else c_out
                _x += self.stride_2
            _y += self.stride_2
        return out_vb 

    
if __name__ == "__main__":
    a = Variable(torch.randn((16, 128, 22, 22)))
    b = Variable(torch.randn((16, 128, 6, 6)))
    c = CorrelationLayer()
    d = c.forward(a, b)
    print(d.shape)