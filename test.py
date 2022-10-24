"""
# =============================================================================
# Code for training - Unimodal  + Multimodal 
#
#Tomé Albuquerque
# =============================================================================
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--architecture', choices=['inception_v3', 'mnasnet1_0', 'mobilenet_v2', 'resnet18',
    'resnext50_32x4d', 'vgg16','wide_resnet50_2'],default='resnext50_32x4d')
parser.add_argument('--method', choices=['UniMRI','MultiMRI'], default='UniMRI')
parser.add_argument('--MRI_type', choices=['flair', 't1', 't1ce', 't2', 'all'], default='flair')
parser.add_argument('--fold', type=int, choices=range(10), default=0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
args = parser.parse_args()

import numpy as np
from time import time
from tqdm import tqdm
from torch import optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import torch
import mydataset, mymodels
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tr_ds = mydataset.MyDataset_MRI('train', mydataset.train_transforms, args.fold, args.MRI_type)
tr = DataLoader(tr_ds, args.batchsize, True, pin_memory=True)
ts_ds = mydataset.MyDataset_MRI('test', mydataset.val_transforms, args.fold, args.MRI_type)
ts = DataLoader(ts_ds, args.batchsize, pin_memory=True)



prefix = '-'.join(f'{k}-{v}' for k, v in vars(args).items())

model = getattr(mymodels, args.method)(args.architecture)

model = model.to(device)
opt = optim.Adam(model.parameters(), args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True)
# np.savetxt('output-' + prefix + '-proba.txt', predict_proba(ts), delimiter=',')


torch.load('weights//architecture-resnext50_32x4d-method-UniMRI-MRI_type-flair-fold-0-epochs-10-batchsize-32-lr-0.0001.pth')
# =============================================================================
# Print some metrics and save in 
# =============================================================================


def predict_metrics(data):
    model.eval()
    Phat = []
    Y_true=[]
    with torch.no_grad():
        for XX, Y in data:
            XX = [X.to(device, torch.float) for X in XX]
            if args.MRI_type in ('flair','t1','t1ce','t2'):
                XX=XX[0]
            
            Y = Y.to(device, torch.float)
            Yhat = model(XX)
            Phat += list(model.to_proba(Yhat).cpu().numpy())
            Y_true += list(Y.cpu().numpy())
    return Y_true, Phat


from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, recall_score, precision_score,roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle

data_test = DataLoader(ts_ds, 1,False,  pin_memory=True)
Y_true, Phat_p = predict_metrics(data_test)

Phat = [Phat_p[i].argmax(0) for i in range(len(Phat_p))]
Phatt=[max(Phat_p[i]) for i in range(len(Phat_p))]

accuracy = accuracy_score(Y_true, Phat)
mae = mean_absolute_error(Y_true, Phat)
f1 = f1_score(Y_true, Phat, average='weighted')
recall = recall_score(Y_true, Phat, average='weighted')
precision = precision_score(Y_true, Phat, average='weighted')
auc = roc_auc_score(Y_true,Phat_p, multi_class='ovr')
os.makedirs("results", exist_ok=True)

f = open('results\\'+ str(prefix)+'.txt', 'a+')
f.write('\n\nModel:'+str(prefix)+
    ' \naccuracy:'+ str(accuracy)+
    ' \nmae:'+ str(mae)+
    ' \nf1:'+str(f1)+
    ' \nrecall:'+ str(recall)+
    ' \nprecision:'+ str(precision)+
    ' \nauc:'+ str(auc))
f.close()



y_test = Y_true

# Binarize the output
y_test = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test.shape[1]


y_score = np.array(Phat_p)

fpr = dict()
tpr = dict()
roc_auc = dict()
lw=2
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])
colors = cycle(['blue', 'red', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC curve for MRI: {args.MRI_type} - Fold: {args.fold}')
plt.legend(loc="lower right")
plt.savefig(f"ROC curve for MRI: {args.MRI_type} - Fold: {args.fold}.png")
plt.show()