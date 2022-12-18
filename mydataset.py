"""
# =============================================================================
# Code for dataset loading - Unimodal dataloaders + Multimodal Dataloader
#
#Tomé Albuquerque
# =============================================================================
"""
from __future__ import print_function, division
from torch.utils.data import Dataset
import torch
import pandas as pd
import pickle
import nibabel as nib
import os
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, ScaleIntensity


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class MRIDataset(Dataset):
    """Brain Tumor MRI Dataset."""

    def __init__(self, fold, modality, dataset_type, transform=None, base_dir='data_multimodal_tcga',
                 pickle_file='modified_multimodal_glioma_data.pickle'):
        """
        Args:
            fold (int): Fold from cross validation. Options: [0, 1, 2, 3, 4]
            modality (string): Image modality of the sample. Block is the cropped image from the segmentation map of
                               region on interest (glioma). Option: ['t1', 't1ce', 't2', 'flair', 't1_block',
                               't1ce_block', 't2_block', 'flair_block', 'all']
            dataset_type (string): Label that specifies whether samples belong to train set or test set.
                                   Option: ['train', 'test']
            transform (Optional, list of callable): List of transform
            base_dir (string): Directory with image folder.
            pickle_file (string): Pickle file that specifies image paths.
        """
        self.transform = transform
        self.fold = fold
        self.modality = modality
        self.dataset_type = dataset_type

        self.base_dir = base_dir
        self.pickle_file = pickle_file

        self.images, self.labels = pd.read_pickle(os.path.join(self.base_dir, self.pickle_file))[self.fold][self.dataset_type]

    def __getitem__(self, i):
        mods = [self.modality]
        if self.modality == 'all':
            mods = ['flair', 'flair_block',
                    't1','t1_block',
                    't1ce', 't1ce_block',
                    't2', 't2_block']
        image = []
        for mod in mods:
            # TODO: a quick hack, will delete later after refactoring
            img_path = self.images[i][mod].replace('\\', os.sep)
            img = nib.load(os.path.join(self.base_dir, img_path)).get_fdata()
            if self.transform:
                img = self.transform(img)
            image.append(torch.moveaxis(img[0], 2, 0))

        # encoding target from ([0, 0], [0, 1], [1, 1]) to (0, 1, 2)
        label = sum([self.labels[i]['idh1'], self.labels[i]['ioh1p15q']])

        return image, label

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), RandRotate90()])
    test_transforms = Compose([ScaleIntensity(), EnsureChannelFirst()])

    ds = MRIDataset(fold=0, modality='flair', dataset_type='train', transform=train_transforms)
    print(ds.__len__())

    ds1 = MRIDataset(fold=0, modality='flair', dataset_type='test', transform=test_transforms)
    train_dataloader = DataLoader(ds, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(ds1, batch_size=2, shuffle=True)
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--type', choices=['train', 'test'],default='train')
    # args = parser.parse_args()

    # import matplotlib.pyplot as plt
    # ds = MyDataset_MRI(args.type, train_transforms, 0)
    # X, Y = ds[0]
    # print('X:', X.min(), X.max(), X.shape, X.dtype)

