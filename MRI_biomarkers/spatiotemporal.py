import torch
import torch.nn as nn
from torch import utils
from torch.utils.data import dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.multiprocessing
import torchvision
import torchio
from torchio.transforms import (
    RescaleIntensity,
    Compose,
)
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.metrics import confusion_matrix
from dataset import MRIDatasets


class R2Plus1dStem4MRI(nn.Sequential):
    """R(2+1)D stem is different from the default one as it uses separated 3D convolution
    """

    def __init__(self):
        super(R2Plus1dStem4MRI, self).__init__(
            nn.Conv3d(1, 155, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3),
                      bias=False),
            nn.BatchNorm3d(155),
            nn.ReLU(inplace=True),

            nn.Conv3d(155, 64, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0),
                      bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))


def train():
    # Transforms
    rescale = RescaleIntensity((0.05, 99.5))
    randaffine = torchio.RandomAffine(scales=(0.9,1.2),degrees=10, isotropic=True, image_interpolation='nearest')
    flip = torchio.RandomFlip(axes=('LR'), p=0.5)
    transforms = [rescale, flip, randaffine]

    transform = Compose(transforms)
    total_samples = MRIDatasets(dataset_path='../data_multimodal_tcga/Radiology',
                          metadata_path='../data_multimodal_tcga/patient-info-tcga.csv').tcga()
    subjects_dataset = torchio.SubjectsDataset(total_samples, transform=transform)

    # train/test split
    train_set_samples = (int(len(total_samples) - 0.2 * len(total_samples)))
    test_set_samples = (int(len(total_samples)) - (train_set_samples))

    trainset, testset = torch.utils.data.random_split(subjects_dataset, [train_set_samples, test_set_samples],
                                                      generator=torch.Generator().manual_seed(55))

    trainloader = DataLoader(dataset=trainset, batch_size=1, shuffle=True, num_workers=1)
    testloader = DataLoader(dataset=testset, batch_size=1, shuffle=True, num_workers=1)

    model = torchvision.models.video.r2plus1d_18(pretrained=False)
    model.stem = R2Plus1dStem4MRI()

    # regularization
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 3)
    )
    # # print(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # for dataset being unbalanced for classes [0, 1, 2]
    class_weights = torch.FloatTensor([1, 2.2, 4.1]).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    # Initialize the prediction and label lists(tensors) for confusion matrix
    predlist = torch.zeros(0, dtype=torch.long).to(device)
    lbllist = torch.zeros(0, dtype=torch.long).to(device)

    # # if load_model:
    # #     the_model = torch.load(Path(os.getcwd(), 'outputs'))
    epochs = 2
    for epoch in range(epochs):

        logs = {}
        total_correct = 0
        total_loss = 0
        total_images = 0
        total_val_loss = 0

        # if epoch % 5 == 0:
        #     checkpoint = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        #     print("Load True: saving checkpoint")
        #     torch.save(model.state_dict(), Path(os.getcwd(), 'outputs'))

            # else:
            #     checkpoint = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
            #                   'optimizer': optimizer.state_dict()}
            #     print("Loade False: saving checkpoint")
            #     save_checkpoint(checkpoint)

        for i, traindata in enumerate(trainloader):
            # print(traindata)
            images = F.interpolate(traindata['t1'][torchio.DATA], scale_factor=(0.7, 0.7, 0.7)).to(device)
            labels = traindata['label'].to(device)
            optimizer.zero_grad()

            # Forward propagation
            outputs = model(images)

            loss = criterion(outputs, labels)

            # Backward prop
            loss.backward()

            # Updating gradients
            optimizer.step()
            # scheduler.step()

            # Total number of labels
            total_images += labels.size(0)

            # Obtaining predictions from max value
            _, predicted = torch.max(outputs.data, 1)

            # Calculate the number of correct answers
            correct = (predicted == labels).sum().item()

            total_correct += correct
            total_loss += loss.item()

            running_trainacc = ((total_correct / total_images) * 100)

            logs['log loss'] = total_loss / total_images
            logs['Accuracy'] = ((total_correct / total_images) * 100)

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
              .format(epoch + 1, epochs, i + 1, len(trainloader), (total_loss / total_images),
                      (total_correct / total_images) * 100))

        # Testing the model

        with torch.no_grad():
            correct = 0
            total = 0

            for testdata in testloader:
                images = F.interpolate(testdata['t1'][torchio.DATA], scale_factor=(0.7, 0.7, 0.7)).to(device)

                labels = testdata['label'].to(device)
                outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)

                predlist = torch.cat([predlist, predicted.view(-1)])  # Append batch prediction results

                lbllist = torch.cat([lbllist, labels.view(-1)])

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                total_losss = loss.item()

                accuracy = correct / total

            print('Test Accuracy of the model: {} %'.format(100 * correct / total))

            logs['val_' + 'log loss'] = total_loss / total
            validationloss = total_loss / total

            validationacc = ((correct / total) * 100)
            logs['val_' + 'Accuracy'] = ((correct / total) * 100)

        # Computing metrics:

    conf_mat = confusion_matrix(lbllist.cpu().numpy(), predlist.cpu().numpy())

    print(conf_mat)
    cls = ["0", "1", "2"]
    # Per-class accuracy
    class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)
    print(class_accuracy)
    plt.figure(figsize=(10, 10))
    # plot_confusion_matrix(conf_mat, cls)
    plt.show()


if __name__ == "__main__":
    train()