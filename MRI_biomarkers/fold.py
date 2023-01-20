import os
import random
import pandas as pd
import numpy as np
import torch
import monai
from monai.data import ImageDataset, DataLoader
from monai.optimizers import LearningRateFinder
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def _get_images_labels(modality, dataset_type='train', fold=0):
    data_path = 'deep-multimodal-glioma-prognosis/data_multimodal_tcga'
    images, labels = pd.read_pickle(os.path.join(data_path, 'modified_multimodal_glioma_data.pickle'))[fold][
        dataset_type]

    ims, lbs = [], []

    for mod in modality:
        if '_block' in mod:
            ims += [os.path.join(data_path, f[mod]) for f in images]
        else:
            ims += [os.path.join(data_path, f[mod].replace('\\', os.sep)) for f in images]

        lbs += [sum([label['idh1'], label['ioh1p15q']]) for label in labels]

    return ims, np.array(lbs, dtype=np.int64)


def _get_dataloader(modality, dataset_type, fold, batch_size, resize_ps=96):
    images, labels = _get_images_labels(modality, dataset_type, fold=fold)
    reader = 'NumpyReader' if all(['_block' in mod for mod in modality]) else None
    if dataset_type == 'train':
        tf = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((resize_ps, resize_ps, resize_ps)), RandRotate90()])
    else:
        tf = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((resize_ps, resize_ps, resize_ps))])
    ds = ImageDataset(image_files=images, labels=labels, transform=tf, reader=reader)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2,
                              pin_memory=torch.cuda.is_available())
    return dataloader


def main(modality, verbose=False):
    
    reader = 'NumpyReader' if all(['_block' in mod for mod in modality]) else None
    fold = 1
    batch_size = 16
    max_epochs = 5
    lower_lr = 5e-5
    num_classes = 3

    train_loader =  _get_dataloader(modality, 'train', fold, batch_size)

    print('Train dataloder:')
    for batch in train_loader:
        print(batch)

    # images, labels = get_images_labels(modality=modality, dataset_type='test', fold=fold)
    # val_ds = ImageDataset(image_files=images, labels=labels, transform=val_transforms, reader=reader)
    # val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=torch.cuda.is_available())

    # print('Val dataloder:')
    # for batch in val_loader:
    #     print(batch)

    # images, labels = get_images_labels(modality=[random.choice(modality)], dataset_type='test', fold=fold)
    # images, labels = get_images_labels(modality=modality, dataset_type='test', fold=fold)
    # test_ds = ImageDataset(image_files=images, labels=labels, transform=val_transforms, reader=reader)
    # test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=2,
    #                              pin_memory=torch.cuda.is_available())
    # print('Test dataloder:')
    # for batch in test_dataloader:
    #     print(batch)

if __name__ == "__main__":
    main(modality=['t1ce_block'], verbose=True)