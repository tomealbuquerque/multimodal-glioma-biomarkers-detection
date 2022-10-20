"""
# =============================================================================
# Code for dataset loading - Unimodal dataloaders + Multimodal Dataloader
#
#Tomé Albuquerque
# =============================================================================
"""
from torch.utils.data import Dataset
from torchvision import models, transforms
import torch
import pickle
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import os
import monai
from monai.data import ImageDataset, DataLoader
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity



class MyDataset_MRI(Dataset):
    def __init__(self, type, transform, fold, MRI_exam):
        self.X, self.Y = pickle.load(open(f'data_multimodal_tcga/multimodal_glioma_data.pickle', 'rb'))[fold][type]
        self.transform = transform
        self.MRI_exam=MRI_exam
    def __getitem__(self, i):
        
        if self.MRI_exam =='flair':
            img = nib.load(os.path.join('data_multimodal_tcga',self.X[i]['flair']))
            img_flair = img.get_fdata()
            X_flair = self.transform(img_flair)
            XX = [torch.moveaxis(X_flair[0],2,0)]
            
        elif self.MRI_exam =='t1':
            img = nib.load(os.path.join('data_multimodal_tcga',self.X[i]['t1']))
            img_t1 = img.get_fdata()
            X_t1 = self.transform(img_t1)
            XX = [torch.moveaxis(X_t1[0],2,0)]

        elif self.MRI_exam =='t1ce':
            img = nib.load(os.path.join('data_multimodal_tcga',self.X[i]['t1ce']))
            img_t1ce = img.get_fdata()
            X_t1ce = self.transform(img_t1ce)
            XX = [torch.moveaxis(X_t1ce[0],2,0)]
            
        elif self.MRI_exam =='t2':
            img = nib.load(os.path.join('data_multimodal_tcga',self.X[i]['t2']))
            img_t2 = img.get_fdata()
            X_t2 = self.transform(img_t2)
            XX = [torch.moveaxis(X_t2[0],2,0)]
         
        else:
            
            img = nib.load(os.path.join('data_multimodal_tcga',self.X[i]['flair']))
            img_flair = img.get_fdata()
            img = nib.load(os.path.join('data_multimodal_tcga',self.X[i]['t1']))
            img_t1 = img.get_fdata()
            img = nib.load(os.path.join('data_multimodal_tcga',self.X[i]['t1ce']))
            img_t1ce = img.get_fdata()
            img = nib.load(os.path.join('data_multimodal_tcga',self.X[i]['t2']))
            img_t2 = img.get_fdata()
            X_flair = self.transform(img_flair)
            X_t1 = self.transform(img_t1)
            X_t1ce = self.transform(img_t1ce)
            X_t2 = self.transform(img_t2)
            
            XX = [torch.moveaxis(X_flair[0],2,0), torch.moveaxis(X_t1[0],2,0), torch.moveaxis(X_t1ce[0],2,0), torch.moveaxis(X_t2[0],2,0)]
            
            
        Y = [self.Y[i]['idh1'], self.Y[i]['ioh1p15q']]
        

        #Encoding targets 
        if Y==[0,0]:
            Y=0
        elif Y==[1,0]:
            Y=1
        elif Y==[0,1]:
            Y=2
        else:
            Y=3
        return XX, Y

    def __len__(self):
        return len(self.X)


train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), RandRotate90()])
val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst()])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['train', 'test'],default='train')
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    ds = MyDataset_MRI(args.type, train_transforms, 0)
    X, Y = ds[0]
    print('X:', X.min(), X.max(), X.shape, X.dtype)

