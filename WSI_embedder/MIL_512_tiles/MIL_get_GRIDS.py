# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 10:25:28 2022

@author: ge92tis
"""

import sys
import os
import pickle
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

parser = argparse.ArgumentParser(description='')
parser.add_argument('--lib', type=str, default='test', help='path to data file')
parser.add_argument('--output', type=str, default='.', help='name of output directory')
parser.add_argument('--fold', type=int, default=1, help='select kfold')
parser.add_argument('--model', type=str, default='checkpoint_best_512_bin_fold_1.pth', help='path to pretrained model')
parser.add_argument('--batch_size', type=int, default=384, help='how many images to sample per slide (default: 100)')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')


# def main():

def inference(loader, model):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        all_probs=[]
        for i, input in enumerate(loader):
            print('Batch: [{}/{}]'.format(i+1, len(loader)))
            input = input.cuda()
            output = F.softmax(model(input), dim=1)
            # print(output)
            all_probs.append(output.cpu().numpy())
            probs[i*args.batch_size:i*args.batch_size+input.size(0)] = output.detach()[:,1].clone()
    return probs.cpu().numpy(), all_probs


def group_max(groups, data, nmax):
    out = np.empty(nmax)
    out[:] = np.nan
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    out[groups[index]] = data[index]
    return list(out)

class MILdataset(data.Dataset):
    def __init__(self, fold, typet, transform):
        
        lib, targets = pickle.load(open(f'../data_multimodal_tcga/multimodal_glioma_data_multi_level.pickle', 'rb'))[fold][typet]


        slide_names=[lib[i]['slide'] for i in range(len(lib))]
        target=[[targets[i]['idh1'], targets[i]['ioh1p19q']] for i in range(len(targets))]
        
                
                #Encoding targets 
        for i in range(len(target)):
            if target[i]==[0,0]:
                target[i]=0
            elif target[i]==[1,0]:
                target[i]=1
            else:
                target[i]=1
        
        
        grids=np.array([lib[i]['tiles_coords'] for i in range(len(lib))], dtype=object)
        # lib = torch.load(libraryfile)
        
        wsi_dir = "/dss/dssfs04/pn25ke/pn25ke-dss-0001/data_multimodal_tcga/Pathology"
         
        slides = []
        for i,name in enumerate(slide_names):
            #print(os.path.join('data',name))
            slides.append(openslide.OpenSlide(os.path.join(wsi_dir,name.split("\\")[1])))
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
        self.mult = 0.5
        self.size = int(np.round(512*1))
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
                img = img.resize((512,512),Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img
        elif self.mode == 2:
            slideIDX, coord, target = self.t_data[index]
            img = self.slides[slideIDX].read_region(coord,self.level,(self.size,self.size)).convert('RGB')
            if self.mult != 1:
                img = img.resize((512,512),Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img, target
        
    def __len__(self):
        if self.mode == 1:
            return len(self.grid)
        elif self.mode == 2:
            return len(self.t_data)

global args
args = parser.parse_args()
#load model

for typee in ['train', 'test']:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #cnn
    model = models.resnet34(True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    ch = torch.load(args.model)
    # model.cuda()
    model= nn.DataParallel(model)
    model.load_state_dict(ch['state_dict'])
    model.to(device)
    print(args.model)
    
    # model = model.cuda()
    cudnn.benchmark = True
    #normalization
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([transforms.ToTensor(),normalize])
    #load data
    dset = MILdataset(typet=typee,fold=args.fold, transform=trans)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    dset.setmode(1)
    probs, all_probs = inference(loader, model)
    # print(all_probs)
    
    arr = np.vstack(all_probs)
    
    np.savetxt(f"predictions_grid_{typee}_fold_{args.fold}_bin.csv", arr, delimiter=",")
    

        
# if __name__ == '__main__':
#     main()