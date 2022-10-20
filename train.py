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
parser.add_argument('--epochs', type=int, default=30)
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

def test(val):
    model.eval()
    val_avg_acc = 0
    for XX, Y in tqdm(val):
        XX = [X.to(device, torch.float) for X in XX]
        if args.MRI_type in ('flair','t1','t1ce','t2'):
            XX=XX[0]

        Y = Y.to(device, torch.int64)
        Yhat = model(XX)
        Khat = model.to_classes(model.to_proba(Yhat))
        val_avg_acc += (Y == Khat).float().mean() / len(val)
    return val_avg_acc

def train(tr, val, epochs=args.epochs, verbose=True):
    for epoch in range(epochs):
        if verbose:
            print(f'* Epoch {epoch+1}/{args.epochs}')
        tic = time()
        model.train()
        avg_acc = 0
        avg_loss = 0
        for XX, Y in tqdm(tr):
            XX = [X.to(device, torch.float) for X in XX]
            if args.MRI_type in ('flair','t1','t1ce', 't2'):
                XX=XX[0]
            Y = Y.to(device, torch.int64)
            opt.zero_grad()
            Yhat = model(XX)
            loss = model.loss(Yhat, Y)
            loss.backward()
            opt.step()
            Khat = model.to_classes(model.to_proba(Yhat))
            avg_acc += (Y == Khat).float().mean() / len(tr)
            avg_loss += loss / len(tr)
        dt = time() - tic
        out = ' - %ds - Loss: %f, Acc: %f' % (dt, avg_loss, avg_acc)
        if val:
            model.eval()
            out += ', Test Acc: %f' % test(val)
        if verbose:
            print(out)
        scheduler.step(avg_loss)

def predict_proba(data):
    model.eval()
    Phat = []
    with torch.no_grad():
        for XX, _ in data:
            XX = [X.to(device, torch.float) for X in XX]
            if args.MRI_type in ('flair','t1','t1ce','t2'):
                XX=XX[0]
            phat = model.to_proba(model(XX))
            Phat += list(phat.cpu().numpy())
    return Phat


prefix = '-'.join(f'{k}-{v}' for k, v in vars(args).items())

model = getattr(mymodels, args.method)(args.architecture)

model = model.to(device)
opt = optim.Adam(model.parameters(), args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True)
train(tr, ts)
# np.savetxt('output-' + prefix + '-proba.txt', predict_proba(ts), delimiter=',')


os.makedirs("weights", exist_ok=True)
torch.save(model.state_dict(), 'weights\\'+str(prefix)+'.pth')

   
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


from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, recall_score, precision_score,roc_auc_score

data_test = DataLoader(ts_ds, 1,False,  pin_memory=True)
Y_true, Phat_p = predict_metrics(data_test)

Phat = [Phat_p[i].argmax(0) for i in range(len(Phat_p))]

accuracy = accuracy_score(Y_true, Phat, normalize=True)
mae = mean_absolute_error(Y_true, Phat)
f1 = f1_score(Y_true, Phat, average='weighted')
recall = recall_score(Y_true, Phat, average='weighted')
precision = precision_score(Y_true, Phat, average='weighted')
# auc = roc_auc_score(Y_true,Phat_p, multi_class='ovr')
os.makedirs("results", exist_ok=True)

f = open('results\\'+ str(prefix)+'.txt', 'a+')
f.write('\n\nModel:'+str(prefix)+
    ' \naccuracy:'+ str(accuracy)+
    ' \nmae:'+ str(mae)+
    ' \nf1:'+str(f1)+
    ' \nrecall:'+ str(recall)+
    ' \nprecision:'+ str(precision))
    # ' \nauc:'+ str(auc))
f.close()