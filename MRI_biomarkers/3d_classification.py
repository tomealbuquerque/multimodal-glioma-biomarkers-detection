import logging
import os

import pandas as pd
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import ImageDataset, DataLoader
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity


def get_images_labels(modality, fold=0, dataset_type='train'):
    data_path = 'deep-multimodal-glioma-prognosis/data_multimodal_tcga'
    images, labels = pd.read_pickle(os.path.join(data_path, 'modified_multimodal_glioma_data.pickle'))[fold][dataset_type]
    if '_block' in modality:        
        images = [os.path.join(data_path, f[modality]+'.npy') for f in images]
    else:
        images = [os.path.join(data_path, f[modality].replace('\\', os.sep)) for f in images]
    labels = np.array([sum([label['idh1'], label['ioh1p15q']]) for label in labels], dtype=np.int64)

    return images, labels


def main():

    modality = 't2_block'
    reader = 'NumpyReader' if '_block' in modality else None

    train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96)), RandRotate90()])
    val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])

    images, labels = get_images_labels(modality=modality, dataset_type='train')
    train_ds = ImageDataset(image_files=images, labels=labels, transform=train_transforms, reader=reader)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    images, labels = get_images_labels(modality=modality, dataset_type='test')
    val_ds = ImageDataset(image_files=images, labels=labels, transform=val_transforms, reader=reader)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=2, pin_memory=torch.cuda.is_available())

    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = monai.networks.nets.EfficientNetBN("efficientnet-b0", spatial_dims=3).to(device)
    model = monai.networks.nets.ViT(in_channels=1, img_size=(96,96,96), patch_size=(16,16,16), pos_embed='conv', classification=True).to(device)
    # model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=3).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)

    # start a typical PyTorch training
    max_epochs = 5
    val_interval = 2
    best_metric = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                num_correct = 0.0
                metric_count = 0
                for val_data in val_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    val_outputs = model(val_images)
                    value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                    metric_count += len(value)
                    num_correct += value.sum().item()
                metric = num_correct / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
                    print("saved new best metric model")
                print(f"current epoch: {epoch+1} current accuracy: {metric:.4f} "
                      f"best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
                writer.add_scalar("val_accuracy", metric, epoch + 1)
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()

    # plt.figure("train", (12, 6))
    # plt.subplot(1, 2, 1)
    # plt.title("Epoch Average Loss")
    # x = [i + 1 for i in range(len(epoch_loss_values))]
    # y = epoch_loss_values
    # plt.xlabel("epoch")
    # plt.plot(x, y)
    # plt.subplot(1, 2, 2)
    # plt.title("Val AUC")
    # x = [val_interval * (i + 1) for i in range(len(metric_values))]
    # y = metric_values
    # plt.xlabel("epoch")
    # plt.plot(x, y)

    # plt.show()



if __name__ == "__main__":
    main()
    # import nibabel as nib
    # path = 'data_multimodal_tcga/Radiology/TCGA-08-0354/flair_block.nii.gz'
    # print(nib.load(path))