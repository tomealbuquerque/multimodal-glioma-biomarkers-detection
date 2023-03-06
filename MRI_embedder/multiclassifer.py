import os
import monai
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from monai.data import ImageDataset, DataLoader
from monai.optimizers import LearningRateFinder
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, mean_absolute_error, \
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

parser = argparse.ArgumentParser(description='Multimodal MRI branch - multiclass classifier')
parser.add_argument('--modalities', nargs='+', type=str, help='modalities for training. Do not mix segmented and '
                                                              'unsegmented modalities. It can contain one or more '
                                                              'modalities.')
parser.add_argument('--fold', type=int, default=0, help='select k-fold')
parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size (default: 128)')
parser.add_argument('--nepochs', type=int, default=100, help='number of epochs')
parser.add_argument('--verbose', type=bool, default=False, help='Print detailed information during training')
parser.add_argument('--MRI_n_outputs', default=3, type=int, help='multiclass vs binary mri')
parser.add_argument('--lower_lr', default=5e-5, type=float, help='lower bound in search of an optimal learning rate')
parser.add_argument('--higher_lr', default=1, type=int, help='higher bound in search of an optimal learning rate. '
                                                             'Note it should be of type int. Default to 1.')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 0)')
parser.add_argument('--resize_ps', default=96, type=int, help='output size of resize transformation')
parser.add_argument('--model_choice', default='denseNet121', choices=['denseNet121', 'efficientNet', 'ViT'],
                    help='Training model, default: denseNet121')


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


def train_and_eval(modality, fold=0, batch_size=16, max_epochs=100, lower_lr=5e-5,
                   higher_lr=1, num_classes=3, resize_ps=96, model_choice='denseNet',
                   verbose=False, num_workers=2):
    reader = 'NumpyReader' if all(['_block' in mod for mod in modality]) else None

    # Different model choices
    if model_choice == 'denseNet121':
        model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=num_classes,
                                                dropout_prob=0.2)
    elif model_choice == 'efficientNet':
        model = monai.networks.nets.EfficientNetBN(model_name="efficientnet-b0", spatial_dims=3,
                                                   in_channels=1, num_classes=num_classes)
    else:
        model = monai.networks.nets.ViT(in_channels=1, img_size=(resize_ps, resize_ps, resize_ps),
                                        patch_size=(16, 16, 16), pos_embed='conv', classification=True,
                                        num_classes=num_classes, post_activation='x')

    train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(),
                                Resize((resize_ps, resize_ps, resize_ps)), RandRotate90()])
    val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((resize_ps, resize_ps, resize_ps))])

    images, labels = get_images_labels(modality=modality, dataset_type='train', fold=fold)
    train_ds = ImageDataset(image_files=images, labels=labels, transform=train_transforms, reader=reader)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=torch.cuda.is_available())

    images, labels = get_images_labels(modality=modality, dataset_type='test', fold=fold)
    val_ds = ImageDataset(image_files=images, labels=labels, transform=val_transforms, reader=reader)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=torch.cuda.is_available())

    if verbose:
        print(f'Train data with {len(train_ds)} images loaded; val data with {len(val_ds)} images loaded!')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # find the optimal learning rate
    optimizer = torch.optim.Adam(model.parameters(), lower_lr)
    loss_function = torch.nn.CrossEntropyLoss()
    lr_finder = LearningRateFinder(model, optimizer, loss_function, device=device)
    lr_finder.range_test(train_loader, val_loader, end_lr=higher_lr, num_iter=20)
    steepest_lr, _ = lr_finder.get_steepest_gradient()

    if verbose:
        print(f'Find the steepest learning rate {steepest_lr}')

    optimizer = torch.optim.Adam(model.parameters(), round(steepest_lr, 5))

    trial_name = '_'.join(modality)
    trial_name += f'_fold{fold}_'
    trial_name += '_'.join([pd.Timestamp.now().strftime('%Y_%m_%d_%X')])
    trial_path = os.path.join(os.getcwd(), "deep-multimodal-glioma-prognosis", "MRI_biomarkers", "results",
                              f"trial_{trial_name}")

    os.makedirs(os.path.join(os.getcwd(), "deep-multimodal-glioma-prognosis", "MRI_biomarkers", "results"),
                exist_ok=True)
    os.makedirs(trial_path, exist_ok=True)

    f = open(os.path.join(trial_path, 'records.txt'), 'a+')

    # start a typical PyTorch training
    best_metric = -1
    epoch_loss_values = []
    loss_tes = []
    metric_values = []
    for epoch in range(max_epochs):
        if verbose:
            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epochs}")
            f.write("-" * 10 + '\n')
            f.write(f"epoch {epoch + 1}/{max_epochs}\n")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            if model_choice == 'ViT':
                loss = loss_function(outputs[0], labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            if verbose:
                print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                f.write(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}\n")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        if verbose:
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
            f.write(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}\n")

        model.eval()
        with torch.no_grad():
            num_correct = 0.0
            metric_count = 0
            val_avg_loss = 0
            for val_data in val_loader:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                val_outputs = model(val_images)
                if model_choice == 'ViT':
                    val_outputs = model(val_images)[0]
                val_loss = loss_function(val_outputs, val_labels)
                value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                metric_count += len(value)
                num_correct += value.sum().item()
                val_avg_loss += val_loss.item() / len(val_loader)
            loss_tes.append(val_avg_loss)
            metric = num_correct / metric_count
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(),
                           os.path.join(trial_path, "best_multiclass_ckpt_best_tiles.pth"))
                print("saved new best metric model")
            if verbose:
                print(f"current epoch: {epoch + 1} current accuracy: {metric:.4f} "
                      f"best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
                f.write(f"current epoch: {epoch + 1} current accuracy: {metric:.4f} "
                        f"best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}\n")
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    f.write(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}\n\n")

    # Evaluate
    images, labels = get_images_labels(modality=[random.choice(modality)], dataset_type='test', fold=fold)
    test_ds = ImageDataset(image_files=images, labels=labels, transform=val_transforms, reader=reader)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=num_workers,
                                 pin_memory=torch.cuda.is_available())

    model.load_state_dict(torch.load(os.path.join(trial_path, "best_multiclass_ckpt_best_tiles.pth")))
    model.eval()

    y_true = []
    y_pred = []
    y_pred_prob = []

    with torch.no_grad():
        for test_data in test_dataloader:
            test_images, test_labels = test_data[0].to(device), test_data[1].to(device)

            test_outputs = model(test_images)
            y_pred_prob.append(nn.Softmax(dim=1)(test_outputs).cpu().numpy()[0])
            test_outputs = test_outputs.argmax(dim=1)
            if model_choice == 'ViT':
                test_outputs = model(test_images)[0].argmax(dim=1)
            for i in range(len(test_outputs)):
                y_true.append(test_labels[i].item())
                y_pred.append(test_outputs[i].item())

    print(classification_report(y_true, y_pred, target_names=['0', '1', '2'], digits=4))

    msg = f"""
    Model: {model.__str__().split('(')[0]}
        number of classes: {num_classes}
        batch size: {batch_size}
        epochs: {max_epochs}
        fold: {fold}
        modality used: {modality}
        trained on {len(train_ds)} images
        evaluates on {len(val_ds)} images from test set.

    Metrics: 
    accuracy: {accuracy_score(y_true, y_pred)}
    balanced accuracy: {balanced_accuracy_score(y_true, y_pred)}
    mae: {mean_absolute_error(y_true, y_pred)}
    auc: {roc_auc_score(y_true, np.vstack(y_pred_prob), multi_class='ovr')}

    Classfication report:
    {classification_report(y_true, y_pred, target_names=['0', '1', '2'], digits=4)}

    Model: {model}

    Optimizer: {optimizer}    
    """

    f.write(msg)
    f.close()

    fig = plt.figure(num='_'.join(modality), figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.plot(loss_tes)
    plt.legend(['train', 'test'], loc='upper left')
    plt.subplot(1, 2, 2)
    plt.title("Val AUC")
    x = [(i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.plot(x, y)
    plt.xlabel("epoch")
    plt.savefig(os.path.join(trial_path, 'loss_auc.png'))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])
    disp.plot()
    plt.savefig(os.path.join(trial_path, 'confusion_matrix.png'))

    print('Finished!')


def main():
    global args
    args = parser.parse_args()
    train_and_eval(modality=args.modalities, fold=args.fold, batch_size=args.batch_size, max_epochs=args.nepochs,
                   lower_lr=args.lower_lr, higher_lr=args.higher_lr, num_classes=args.MRI_n_outputs,
                   resize_ps=args.resize_ps, model_choice=args.model_choice, verbose=args.verbose,
                   num_workers=args.workers)


if __name__ == "__main__":
    main()
