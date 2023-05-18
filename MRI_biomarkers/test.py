import os
import random
import pandas as pd
import numpy as np
import torch
import monai
import torch.nn as nn
from monai.data import ImageDataset, DataLoader
from monai.optimizers import LearningRateFinder
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, mean_absolute_error, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def get_images_labels(modality, fold=0, dataset_type='train', data_type='epic'):
    root_dir = 'deep-multimodal-glioma-prognosis'
    data_path = 'data_tum_epic' if data_type == 'epic' else 'data_multimodal_tcga'
    pickle_file = 'multimodal_glioma_data.pickle' if data_type == 'epic' else 'modified_multimodal_glioma_data.pickle'
    pickle_path = os.path.join(root_dir, data_path, pickle_file)
    # codeletion column name is different in csv. TUM: 1p19q; TCGA: ioh1p15q
    codeletion = '1p19q' if data_type == 'epic' else 'ioh1p15q'

    images, labels = pd.read_pickle(pickle_path)[fold][dataset_type]

    ims, lbs = [], []

    for mod in modality:
        if data_type == 'epic':
            ims += [x[mod] for x in images]
            
        else:
            if '_block' in mod:
                ims += [os.path.join(root_dir, data_path, f[mod]) for f in images]
            else:
                ims += [os.path.join(root_dir, data_path, f[mod].replace('\\', os.sep)) for f in images]
        
        lbs += [lb['idh1']+lb[codeletion] for lb in labels]
    
    return ims, np.array(lbs, dtype=np.int64), ims


def main(test_modality, ckpt_path, fold=0, verbose=False):
    resize_ps = 96

    reader = 'NumpyReader' if all(['_block' in mod for mod in test_modality]) else None
    batch_size = 16
    max_epochs = 100
    lower_lr = 5e-5
    num_classes = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(ckpt_path)
    p = ckpt_path.split(os.sep)[-1].split('.')[0]
    test_mod = '_'.join(test_modality)

    f = open(os.path.join(os.getcwd(), "deep-multimodal-glioma-prognosis", "MRI_biomarkers", "results", f'{p}_on_{test_mod}_fold_{fold}.csv'), 'w')
    # model = monai.networks.nets.EfficientNetBN(model_name="efficientnet-b0", spatial_dims=3, in_channels=1, num_classes=num_classes)
    # model = monai.networks.nets.ViT(in_channels=1, img_size=(96,96,96), patch_size=(16,16,16), pos_embed='conv', classification=True, num_classes=num_classes, post_activation='x')
    
    f.write('file,target,prediction,probability_0,probability_1,probability_2\n')
    
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=num_classes, dropout_prob=0.2)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    model = model.to(device)
    val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((resize_ps, resize_ps, resize_ps))])
    # Evaluate
    test_images_tcga, test_labels_tcga, file_names_tcga = get_images_labels(modality=test_modality, dataset_type='test', fold=fold, data_type='tcga')
    test_images_epic, test_labels_epic, file_names_epic = get_images_labels(modality=test_modality, dataset_type='test', fold=fold, data_type='epic')
    test_images = test_images_tcga + test_images_epic
    test_labels = np.concatenate((test_labels_tcga, test_labels_epic), axis=None)
    file_names = file_names_tcga + file_names_epic

    test_ds = ImageDataset(image_files=test_images, labels=test_labels, transform=val_transforms, reader=reader)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    y_true = []
    y_pred = []
    y_pred_prob = []
    with torch.no_grad():
        
        for idx, test_data in enumerate(test_dataloader):
            test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
            test_outputs = model(test_images)
            y_pred_prob.append(nn.Softmax(dim=1)(test_outputs).cpu().numpy()[0])
            
            probs = nn.Softmax(dim=1)(test_outputs)[0]
            test_outputs = test_outputs.argmax(dim=1)
            f.write(f'{file_names[idx]},{test_labels.detach().cpu().numpy()[0]},{test_outputs.detach().cpu().numpy()[0]},{probs[0].item()},{probs[1].item()},{probs[2].item()}\n')
            for i in range(len(test_outputs)):
                y_true.append(test_labels[i].item())
                y_pred.append(test_outputs[i].item())

    
    f.close()

    print('Finished!')


if __name__ == "__main__":
    
    for i in range(5):
        ckpt_path = f'deep-multimodal-glioma-prognosis/MRI_biomarkers/results/train_on_epic&tcga/t1ce_block_flair_block_fold{i}/best_multiclass_ckpt_best_tiles.pth'
        main(test_modality=['t1ce_block', 'flair_block'], ckpt_path=ckpt_path, fold=i, verbose=True)
        ckpt_path = f'deep-multimodal-glioma-prognosis/MRI_biomarkers/results/train_on_epic&tcga/t1ce_block_fold{i}/best_multiclass_ckpt_best_tiles.pth'
        main(test_modality=['t1ce_block'], ckpt_path=ckpt_path, fold=i, verbose=True)
        ckpt_path = f'deep-multimodal-glioma-prognosis/MRI_biomarkers/results/train_on_epic&tcga/t1ce_flair_fold{i}/best_multiclass_ckpt_best_tiles.pth'
        main(test_modality=['t1ce', 'flair'], ckpt_path=ckpt_path, fold=i, verbose=True)
        ckpt_path = f'deep-multimodal-glioma-prognosis/MRI_biomarkers/results/train_on_epic&tcga/t1ce_fold{i}/best_multiclass_ckpt_best_tiles.pth'
        main(test_modality=['t1ce'], ckpt_path=ckpt_path, fold=i, verbose=True)
