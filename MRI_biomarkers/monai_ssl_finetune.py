import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from monai.inferers import sliding_window_inference
from monai.config import print_config
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity

from monai.networks.nets import UNETR
from monai.data import ImageDataset, DataLoader


def get_images_labels(modality, fold=0, dataset_type='train'):
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


if __name__ == "__main__":

    trial_name = '_'.join([pd.Timestamp.now().strftime('%Y_%m_%d_%X')])
    logdir_path = os.path.join('deep-multimodal-glioma-prognosis/MRI_biomarkers/ssl_results_ft', trial_name)
    os.makedirs(logdir_path, exist_ok=True)
    
    use_pretrained = True
    pretrained_path = os.path.normpath('deep-multimodal-glioma-prognosis/MRI_biomarkers/ssl_results/2023_01_27_02:44:08/best_model.pt')
    vit_dict = torch.load(pretrained_path)
    vit_weights = vit_dict["state_dict"]
    print(vit_weights)

    # Training Hyper-params
    lr = 1e-4
    eval_num = 100
    resize_ps = 96
    reader = 'NumpyReader' if all(['_block' in mod for mod in modality]) else None
    
    # Transforms
    train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((resize_ps, resize_ps, resize_ps)), RandRotate90()])
    val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((resize_ps, resize_ps, resize_ps))])
    
    images, labels = get_images_labels(modality=modality, dataset_type='train', fold=fold)
    train_ds = ImageDataset(image_files=images, labels=labels, transform=train_transforms, reader=reader)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2,
                              pin_memory=torch.cuda.is_available())

    images, labels = get_images_labels(modality=modality, dataset_type='test', fold=fold)
    val_ds = ImageDataset(image_files=images, labels=labels, transform=val_transforms, reader=reader)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=torch.cuda.is_available())

    
    