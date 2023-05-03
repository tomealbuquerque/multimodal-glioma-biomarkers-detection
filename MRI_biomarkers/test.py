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

    return ims, np.array(lbs, dtype=np.int64), images


def main(test_modality, ckpt_path, fold=0, verbose=False):
    resize_ps = 96

    modality = ['t1ce_block', 'flair_block']
    reader = 'NumpyReader' if all(['_block' in mod for mod in modality]) else None
    batch_size = 16
    max_epochs = 100
    lower_lr = 5e-5
    num_classes = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(ckpt_path)
    p = ckpt_path.split(os.sep)[-1].split('.')[0]
    test_mod = '_'.join(test_modality)

    f = open(os.path.join(os.getcwd(), "deep-multimodal-glioma-prognosis", "MRI_biomarkers", "results", f'{p}_on_{test_mod}.csv'), 'w')
    # model = monai.networks.nets.EfficientNetBN(model_name="efficientnet-b0", spatial_dims=3, in_channels=1, num_classes=num_classes)
    # model = monai.networks.nets.ViT(in_channels=1, img_size=(96,96,96), patch_size=(16,16,16), pos_embed='conv', classification=True, num_classes=num_classes, post_activation='x')
    
    f.write('file,target,prediction,probability_0,probability_1,probability_2\n')
    
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=num_classes, dropout_prob=0.2)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    model = model.to(device)
    val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((resize_ps, resize_ps, resize_ps))])
    # Evaluate
    images, labels, file_names = get_images_labels(modality=test_modality, dataset_type='test', fold=fold)
    test_ds = ImageDataset(image_files=images, labels=labels, transform=val_transforms, reader=reader)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

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
                f.write(f'{file_names[idx]},{test_labels.detach().cpu().numpy()[0]},{test_outputs.detach().cpu().numpy()[0]},{probs[0].item()},{probs[1].item()},{probs[2].item()}\n')
            else:
                f.write(f'{file_names[idx][test_modality[0]]},{test_labels.detach().cpu().numpy()[0]},{test_outputs.detach().cpu().numpy()[0]},{probs[0].item()},{probs[1].item()},{probs[2].item()}\n')
            
            # test_outputs = model(test_images)[0].argmax(dim=1) # ViT
            for i in range(len(test_outputs)):
                y_true.append(test_labels[i].item())
                y_pred.append(test_outputs[i].item())

    # msg = f"""
    # Eval: {ckpt_path}

    # Model: {model.__str__().split('(')[0]}
    # batch size: {batch_size}
    # epochs: {max_epochs}
    # fold: {fold}
    # evaluates on modality {test_modality} with {len(test_ds)} images.
    
    # Metrics: 
    # accuracy: {accuracy_score(y_true, y_pred)}
    # balanced accuracy: {balanced_accuracy_score(y_true, y_pred)}
    # mae: {mean_absolute_error(y_true, y_pred)}
    # auc: {roc_auc_score(y_true, np.vstack(y_pred_prob), multi_class='ovr')}

    # Classfication report:
    # {classification_report(y_true, y_pred, target_names=['0', '1', '2'], digits=4)}
    # """

    # f.write(msg)
    f.close()

    print('Finished!')


if __name__ == "__main__":
    
    for idx in range(5):
        for tm in [['t1ce'], ['flair'], ['t1ce', 'flair']]:
            
            path = f'deep-multimodal-glioma-prognosis/MRI_biomarkers/saved_ckpt/orig_t1ce_flair_fold{idx}.pth'
            # path = f'deep-multimodal-glioma-prognosis/MRI_biomarkers/saved_ckpt/multiclass_fold{idx}_t1ce_flair.pth'
            main(test_modality=tm, ckpt_path=path, fold=idx, verbose=True)
            # break
            path = f'deep-multimodal-glioma-prognosis/MRI_biomarkers/saved_ckpt/orig_t1ce_fold{idx}.pth'
            # path = f'deep-multimodal-glioma-prognosis/MRI_biomarkers/saved_ckpt/multiclass_fold{idx}_t1ce.pth'
            main(test_modality=tm, ckpt_path=path, fold=idx, verbose=True)
        # break