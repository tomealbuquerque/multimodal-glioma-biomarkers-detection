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
    data_path = 'deep-multimodal-glioma-prognosis/data_tum_epic'
    images, labels = pd.read_pickle(os.path.join(data_path, 'multimodal_glioma_data.pickle'))[fold][
        dataset_type]

    ims, lbs = [], []

    for mod in modality:
        # if '_block' in mod:
        #     ims += [os.path.join(data_path, f[mod]) for f in images]
        # else:
        #     ims += [os.path.join(data_path, f[mod].replace('\\', os.sep)) for f in images]

        # codeletion column name is different in csv. TUM: 1p19q; TCGA: ioh1p15q
        codeletion = '1p19q' if data_type == 'epic' else 'ioh1p15q'
        # lbs += [sum([label['idh1'], label[codeletion]]) for label in labels]

        ims += [x[mod] for x in images]
        lbs += [lb['idh1']+lb[codeletion] for lb in labels]

    return ims, np.array(lbs, dtype=np.int64), ims


def main(modality, fold, verbose=False):
    resize_ps = 96

    # modality = ['flair_block', 't1ce_block']
    reader = 'NumpyReader' if all(['_block' in mod for mod in modality]) else None
    batch_size = 16
    max_epochs = 100
    lower_lr = 5e-5
    num_classes = 3

    # model = monai.networks.nets.EfficientNetBN(model_name="efficientnet-b0", spatial_dims=3, in_channels=1, num_classes=num_classes)
    # model = monai.networks.nets.ViT(in_channels=1, img_size=(96,96,96), patch_size=(16,16,16), pos_embed='conv', classification=True, num_classes=num_classes, post_activation='x')
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=num_classes, dropout_prob=0.2)

    train_transforms = Compose(
        [ScaleIntensity(), EnsureChannelFirst(), Resize((resize_ps, resize_ps, resize_ps)), RandRotate90()])
    val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((resize_ps, resize_ps, resize_ps))])

    images, labels, _ = get_images_labels(modality=modality, dataset_type='train', fold=fold)
    train_ds = ImageDataset(image_files=images, labels=labels, transform=train_transforms, reader=reader)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2,
                              pin_memory=torch.cuda.is_available())

    test_images, test_labels, file_names = get_images_labels(modality=modality, dataset_type='test', fold=fold)
    val_ds = ImageDataset(image_files=test_images, labels=test_labels, transform=val_transforms, reader=reader)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=torch.cuda.is_available())

    if verbose:
        print(f'Train data with {len(train_ds)} images loaded; val data with {len(val_ds)} images loaded!')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lower_lr)
    loss_function = torch.nn.CrossEntropyLoss()
    lr_finder = LearningRateFinder(model, optimizer, loss_function, device=device)
    lr_finder.range_test(train_loader, val_loader, end_lr=1e-0, num_iter=20)
    steepest_lr, _ = lr_finder.get_steepest_gradient()

    if verbose:
        print(f'Find the steepest learning rate {steepest_lr}')

    optimizer = torch.optim.Adam(model.parameters(), round(steepest_lr, 5))

    trial_name = '_'.join(modality)
    path_name = trial_name + f'_fold{fold}_'
    trial_name = path_name + '_'.join([pd.Timestamp.now().strftime('%Y_%m_%d_%X')])
    trial_path = os.path.join(os.getcwd(), "deep-multimodal-glioma-prognosis", "MRI_biomarkers", "results",
                              f"{trial_name}")

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
            # loss = loss_function(outputs[0], labels) # ViT
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
                # val_outputs = model(val_images)[0] # ViT
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
    f_csv = open(os.path.join(os.getcwd(), "deep-multimodal-glioma-prognosis", "MRI_biomarkers", "results",
                          f'epic_{path_name}.csv'), 'w')
    f_csv.write('file,target,prediction,probability_0,probability_1,probability_2\n')

    model.load_state_dict(torch.load(os.path.join(trial_path, "best_multiclass_ckpt_best_tiles.pth")))
    model.eval()

    y_true = []
    y_pred = []
    y_pred_prob = []

    images, labels, file_names = get_images_labels(modality=modality, dataset_type='test', fold=fold)
    test_ds = ImageDataset(image_files=images, labels=labels, transform=val_transforms, reader=reader)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=2,
                                 pin_memory=torch.cuda.is_available())
    # print('FILE_NAMES: ', file_names)
    # print('TEST_DataLoader', len(test_dataloader))

    with torch.no_grad():
        for idx, test_data in enumerate(test_dataloader):
            test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
            
            test_outputs = model(test_images)
            y_pred_prob.append(nn.Softmax(dim=1)(test_outputs).cpu().numpy()[0])
            
            probs = nn.Softmax(dim=1)(test_outputs)[0]
            test_outputs = test_outputs.argmax(dim=1)
            # test_outputs = model(test_images)[0].argmax(dim=1) # ViT
            # if len(modality) <= 1:
            #     f_csv.write(
            #         f'{file_names[idx]},{test_labels.detach().cpu().numpy()[0]},{test_outputs.detach().cpu().numpy()[0]},{probs[0].item()},{probs[1].item()},{probs[2].item()}\n')
            # else:
                # f_csv.write(f'{file_names[idx][modality[0]]},{test_labels.detach().cpu().numpy()[0]},{test_outputs.detach().cpu().numpy()[0]},{probs[0].item()},{probs[1].item()},{probs[2].item()}\n')
            f_csv.write(f'{file_names[idx]},{test_labels.detach().cpu().numpy()[0]},{test_outputs.detach().cpu().numpy()[0]},{probs[0].item()},{probs[1].item()},{probs[2].item()}\n')
            
            for i in range(len(test_outputs)):
                y_true.append(test_labels[i].item())
                y_pred.append(test_outputs[i].item())
    
    f_csv.close()
    print(classification_report(y_true, y_pred, target_names=['0', '1', '2'], digits=4))

    msg = f"""
    Model: {model.__str__().split('(')[0]}
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
    # print(len(x), len(y))
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.plot(loss_tes)
    # print(len(loss_tes))
    plt.legend(['train', 'test'], loc='upper left')
    plt.subplot(1, 2, 2)
    plt.title("Val AUC")
    x = [(i + 1) for i in range(len(metric_values))]
    y = metric_values
    # print(len(x), len(y))
    plt.plot(x, y)
    plt.xlabel("epoch")
    plt.savefig(os.path.join(trial_path, 'loss_auc.png'))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])
    disp.plot()
    plt.savefig(os.path.join(trial_path, 'confusion_matrix.png'))

    print('Finished!')


if __name__ == "__main__":
    # main(modality=['t1ce', 'flair'], fold=0, verbose=True)
    for i in range(5):
        main(modality=['t1ce', 'flair'], fold=i, verbose=True)
        main(modality=['t1ce'], fold=i, verbose=True)
        main(modality=['t1ce_block', 'flair_block'], fold=i, verbose=True)
        main(modality=['t1ce_block'], fold=i, verbose=True)