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
    def __init__(self, type, transform, fold):
        self.X, self.Y = pickle.load(open(f'data_multimodal_tcga/multimodal_glioma_data.pickle', 'rb'))[fold][type]
        self.transform = transform

    def __getitem__(self, i):
        img = nib.load(os.path.join('data_multimodal_tcga',self.X[i]['flair']))
        img_data = img.get_fdata()
        X = self.transform(img_data)
        
        Y = [self.Y[i]['idh1'], self.Y[i]['ioh1p15q']]
        if Y==[0,0]:
            Y=0
        elif Y==[1,0]:
            Y=1
        elif Y==[0,1]:
            Y=2
        else:
            Y=3
        return torch.moveaxis(X[0],2,0), Y

    def __len__(self):
        return len(self.X)

train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), RandRotate90()])
val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst()])


class MyDataset_MRI_all(Dataset):
    def __init__(self, type, transform, fold):
        self.X, self.Y = pickle.load(open(f'data_multimodal_tcga/multimodal_glioma_data.pickle', 'rb'))[fold][type]
        self.transform = transform

    def __getitem__(self, i):
        
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
        
        Y = [self.Y[i]['idh1'], self.Y[i]['ioh1p15q']]
        
        XX = [torch.moveaxis(X_flair[0],2,0), torch.moveaxis(X_t1[0],2,0), torch.moveaxis(X_t1ce[0],2,0), torch.moveaxis(X_t2[0],2,0)]
        
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

# train_transforms = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.RandomAffine(180, (0, 0.1), (0.9, 1.1)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.ColorJitter(saturation=(0.5, 2.0)),
#     transforms.ToTensor(),  # vgg normalization
# ])

# val_transforms = transforms.Compose([
#     transforms.ToTensor(),  # vgg normalization
# ])



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['train', 'test'],default='train')
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    ds = MyDataset_MRI(args.type, train_transforms, 0)
    X, Y = ds[0]
    print('X:', X.min(), X.max(), X.shape, X.dtype)

