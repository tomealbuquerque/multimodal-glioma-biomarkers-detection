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
            img = torch.from_numpy(img).permute(2, 0, 1)
            image.append(img)

        return image, label

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    import monai
    from monai.data import ImageDataset, DataLoader
    import numpy as np
    ds = MRIDataset(fold=0, modality='flair', dataset_type='train')
    data_path = 'data_multimodal_tcga'
    images = [os.path.join(data_path, f['flair'].replace('\\', os.sep)) for f in ds.images]
    labels = np.array([sum([label['idh1'], label['ioh1p15q']]) for label in ds.labels], dtype=np.int64)

    train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), RandRotate90()])

    # check_ds = ImageDataset(image_files=images, labels=labels, transform=train_transforms)
    # check_loader = DataLoader(check_ds, batch_size=2, num_workers=2, pin_memory=torch.cuda.is_available())
    # im, label = monai.utils.misc.first(check_loader)
    # print(type(im), im.shape, label)

    train_ds = ImageDataset(image_files=images[:10], labels=labels[:10], transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())