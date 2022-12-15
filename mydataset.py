"""
# =============================================================================
# Code for dataset loading - Unimodal dataloaders + Multimodal Dataloader
#
#Tomé Albuquerque
# =============================================================================
"""
from torch.utils.data import Dataset
import torch
import pickle
import nibabel as nib
import os
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, ScaleIntensity


class MRIDataset(Dataset):
    """
    Block is the cropped image from the segmentation map of region on interest (glioma)

    type: str, options ['train', 'test']
    fold: int [0, 4]
    modality: str, options ['t1', 't1ce', 't2', 'flair', 't1_block', 't1ce_block', 't2_block', 'flair_block', 'all']
    """
    def __init__(self, type, transform, fold, modality):
        self.base_ds_dir = 'data_multimodal_tcga'
        pickle_file = os.path.join(self.base_ds_dir, r'modified_multimodal_glioma_data.pickle')
        self.X, self.Y = pickle.load(open(pickle_file, 'rb'))[fold][type]
        self.transform = transform
        self.modality=modality

    def __getitem__(self, i):
        mods = [self.modality]
        if self.modality == 'all':
            mods = ['flair', 'flair_block',
                    't1','t1_block',
                    't1ce', 't1ce_block',
                    't2', 't2_block']
        XX = []
        for mod in mods:
            # TODO: a quick hack, will delete later after refactoring
            img_path = self.X[i][mod].replace('\\', os.sep)
            img = nib.load(os.path.join(self.base_ds_dir, img_path))
            img = img.get_fdata()
            X_ = self.transform(img)
            XX.append(torch.moveaxis(X_[0], 2, 0))

        # encoding target from ([0, 0], [0, 1], [1, 1]) to (0, 1, 2)
        Y = sum([self.Y[i]['idh1'], self.Y[i]['ioh1p15q']])

        return XX, Y

    def __len__(self):
        return len(self.X)


train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), RandRotate90()])
val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst()])


if __name__ == '__main__':
    ds = MRIDataset(type='train', transform=train_transforms, fold=0, modality='flair')
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(ds, batch_size=64, shuffle=True)
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

