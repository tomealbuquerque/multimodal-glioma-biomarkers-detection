# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 15:09:25 2022

@author: ge92tis
"""
import sys
import os
os.add_dll_directory('C:\\Users\\ge92tis\\openslide-win64-20220811\\bin')
import numpy as np
import argparse
import random
import openslide
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset
from torchvision import models, transforms
import pickle
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import os




class MILdataset(data.Dataset):
    def __init__(self, fold, typet, transform):
        
        lib, targets = pickle.load(open(f'data_multimodal_tcga/multimodal_glioma_data.pickle', 'rb'))[0]['train']
        
        slide_names=[lib[i]['slide'] for i in range(len(lib))]
        target=[[targets[i]['idh1'], targets[i]['ioh1p15q']] for i in range(len(targets))]
        grids=np.array([lib[i]['tiles_coords'] for i in range(len(lib))])
        
        # lib = torch.load(libraryfile)
        slides = []
        for i,name in enumerate(slide_names):
            slides.append(openslide.OpenSlide(os.path.join('data_multimodal_tcga/Pathology',name)))
        print('')
        #Flatten grid
        grid = []
        slideIDX = []
        for i,g in enumerate(grids):
            grid.extend(g)
            slideIDX.extend([i]*len(g))

        print('Number of tiles: {}'.format(len(grid)))
        self.slidenames = slide_names
        self.slides = slides
        self.targets = target
        self.grid = grid
        self.slideIDX = slideIDX
        self.transform = transform
        self.mode = None
        self.mult = 1
        self.size = int(np.round(224*1))
        self.level = 0
    def setmode(self,mode):
        self.mode = mode
    def maketraindata(self, idxs):
        self.t_data = [(self.slideIDX[x],self.grid[x],self.targets[self.slideIDX[x]]) for x in idxs]
    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))
    def __getitem__(self,index):
        if self.mode == 1:
            slideIDX = self.slideIDX[index]
            coord = self.grid[index]
            img = self.slides[slideIDX].read_region(coord,self.level,(self.size,self.size)).convert('RGB')
            if self.mult != 1:
                img = img.resize((224,224),Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img
        elif self.mode == 2:
            slideIDX, coord, target = self.t_data[index]
            img = self.slides[slideIDX].read_region(coord,self.level,(self.size,self.size)).convert('RGB')
            if self.mult != 1:
                img = img.resize((224,224),Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img, target
    def __len__(self):
        if self.mode == 1:
            return len(self.grid)
        elif self.mode == 2:
            return len(self.t_data)

normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
trans = transforms.Compose([transforms.ToTensor(), normalize])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['train', 'test'],default='train')
    args = parser.parse_args()

    ds = MILdataset(0, 'train', trans)
    # train_loader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)
    # ds = MILdataset(args.type, aug_transforms, 0)
    X = ds[1]
    # print('X:', X.min(), X.max(), X.shape, X.dtype)

