#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 17:54:35 2018

@author: jlaplaza
"""

import os

import matplotlib.pyplot as plt
from skimage import io

from torch.utils.data import Dataset


class CFNetDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            
            root_dir (string): Directory to ILSVRC root directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.root_dir = root_dir
        self.im_dir = os.path.join(self.root_dir, "crops/ILSVRC/Data/VID/")
        
        self.train_len = 0
        
        self.x_img_list = []
        
        # iterate through train and test
        for dataset in os.listdir(self.im_dir):
            if dataset == "train":
                self.train_dir = os.path.join(self.im_dir, dataset)
                
                # iterate through a, b, c, d and e folders
                for folder in os.listdir(self.train_dir):
                    path = os.path.join(self.train_dir, folder)
                    
                    # iterate through each sequence 
                    for vid in os.listdir(path):
                        vid_path = os.path.join(path, vid)   
                        self.train_len = self.train_len + int(len(os.listdir(vid_path)) / 4)
                        for frame in os.listdir(vid_path):
                            #print(frame[-5:])
                            if frame[-5:] == "x.jpg":
                                img_fname = os.path.join(vid_path, frame)
                                self.x_img_list.append(img_fname)
        #print(self.x_img_list)
                            
                    
        self.transform = transform
        

    def __len__(self):
        return self.train_len
    
    
    def __getitem__(self, idx):
        x = self.x_img_list[idx]
        z = x[:-5] + "z.jpg"
        x = io.imread(x)
        z = io.imread(z)
        
        sample = {'x': x, 'z': z}
        
        return sample
        
    
if __name__ == "__main__":
    
    #Create the dataset object
    dataset = CFNetDataset("/home/jlaplaza/Downloads/ILSVRC2017_VID/")
    
    #Check dataset's lenght
    print("Dataset len: %i" % len(dataset))
    
    #Sanity check of the ith sample
    idx = 10   
    d = dataset[idx]
    #d['x'] = d['x'][...,::-1]
    
    fig=plt.figure(figsize=(8, 8))
    
    print("x image:")
    fig.add_subplot(2, 1, 1)
    plt.imshow(d['x'], aspect='auto')
    
    print("z image:")
    fig.add_subplot(2, 1, 2)
    plt.imshow(d['z'], aspect='auto')
    
    plt.show()
    
    del dataset
        