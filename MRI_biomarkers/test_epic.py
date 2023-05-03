import os
import numpy as np
import pandas as pd
import torch
import monai
import torch.nn as nn
from monai.data import ImageDataset, DataLoader
from monai.transforms import EnsureChannelFirst, Compose, Resize, ScaleIntensity


root_dir = '/dss/dssfs04/pn25ke/pn25ke-dss-0001/albuquerque/Glioma_EPIC'

def save_mri_file_to_dict():

    df = pd.read_csv(os.path.join(root_dir, 'Gliome_EPIC_PhenoData.csv'))

    df = df.dropna(subset=['IDH', 'LOH_1p19q'])
    X, y = [], []

    for root, dirs, files in os.walk(root_dir):
        if 'preop' in root:
            subfolder_id = root.split(os.sep)[7]
            if subfolder_id in df.Pseudonym.values:
                X.append({
                 't1ce_block': os.path.join(root, f'sub-{subfolder_id}_ses-preop_space-sri_t1c.npy'),
                 'flair_block': os.path.join(root, f'sub-{subfolder_id}_ses-preop_space-sri_flair.npy'),
                 'flair': os.path.join(root, f'sub-{subfolder_id}_ses-preop_space-sri_flair.nii.gz'),
                 't1ce':  os.path.join(root, f'sub-{subfolder_id}_ses-preop_space-sri_t1c.nii.gz'),
                 't1': os.path.join(root, f'sub-{subfolder_id}_ses-preop_space-sri_t1.nii.gz'),
                 't2': os.path.join(root, f'sub-{subfolder_id}_ses-preop_space-sri_t2.nii.gz'),
                })

                idx = df[df['Pseudonym'] == subfolder_id]
                y.append({
                    'idh1': int(idx['IDH'].item()),
                    '1p19q': int(idx['LOH_1p19q'].item())
                })
    return X, y


def get_images_labels(modality, X, y):
    ims, lbs = [], []

    for mod in modality:
        ims += [x[mod] for x in X]
        lbs += [lb['idh1']+lb['1p19q'] for lb in y]

    return ims, np.array(lbs, dtype=np.int64), X


def train(test_modality, ckpt_path):
    resize_ps = 96

    reader = 'NumpyReader' if all(['_block' in mod for mod in test_modality]) else None
    num_classes = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(ckpt_path)
    p = ckpt_path.split(os.sep)[-1].split('.')[0]
    test_mod = '_'.join(test_modality)

    f = open(os.path.join(os.getcwd(), "deep-multimodal-glioma-prognosis", "MRI_biomarkers", "results",
                          f'epic_{p}_on_{test_mod}.csv'), 'w')
    f.write('file,target,prediction,probability_0,probability_1,probability_2\n')

    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=num_classes, dropout_prob=0.2)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    model = model.to(device)
    val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((resize_ps, resize_ps, resize_ps))])

    # Evaluate
    X, y = save_mri_file_to_dict()
    print(f'Load data from {root_dir} success! Got {len(X)} images & labels.')
    images, labels, file_names = get_images_labels(test_modality, X, y)
    test_ds = ImageDataset(image_files=images, labels=labels, transform=val_transforms, reader=reader)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=2,
                                 pin_memory=torch.cuda.is_available())

    y_true = []
    y_pred = []
    y_pred_prob = []
    if len(test_modality) > 1:
        files1 = []
        files2 = []
        for file_name in file_names:
            files1.append(file_name[test_modality[0]])
            files2.append(file_name[test_modality[1]])
        file_names = files1 + files2

    with torch.no_grad():

        for idx, test_data in enumerate(test_dataloader):
            test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
            test_outputs = model(test_images)
            # y_pred_prob.append(nn.Softmax(dim=1)(test_outputs).cpu().numpy()[0])

            probs = nn.Softmax(dim=1)(test_outputs)[0]
            test_outputs = test_outputs.argmax(dim=1)

            if len(test_modality) > 1:
                f.write(
                    f'{file_names[idx]},{test_labels.detach().cpu().numpy()[0]},{test_outputs.detach().cpu().numpy()[0]},{probs[0].item()},{probs[1].item()},{probs[2].item()}\n')
            else:
                f.write(
                    f'{file_names[idx][test_modality[0]]},{test_labels.detach().cpu().numpy()[0]},{test_outputs.detach().cpu().numpy()[0]},{probs[0].item()},{probs[1].item()},{probs[2].item()}\n')

            # test_outputs = model(test_images)[0].argmax(dim=1) # ViT
            for i in range(len(test_outputs)):
                y_true.append(test_labels[i].item())
                y_pred.append(test_outputs[i].item())

    f.close()
    print('Finished!')


if __name__ == '__main__':
    
    for i in range(0,5):
        # path = f'deep-multimodal-glioma-prognosis/MRI_biomarkers/saved_ckpt/orig_t1ce_fold{i}.pth'
        path = f'deep-multimodal-glioma-prognosis/MRI_biomarkers/saved_ckpt/multiclass_fold{i}_t1ce_flair.pth'
        train(test_modality=['t1ce_block', 'flair_block'], ckpt_path=path)

        path = f'deep-multimodal-glioma-prognosis/MRI_biomarkers/saved_ckpt/orig_t1ce_flair_fold{i}.pth'
        train(test_modality=['t1ce', 'flair'], ckpt_path=path)