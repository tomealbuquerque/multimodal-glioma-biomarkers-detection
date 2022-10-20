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
parser.add_argument('--method', choices=['Base'], default='Base')
parser.add_argument('--fold', type=int, choices=range(10), default=0)
parser.add_argument('--epochs', type=int, default=50)
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tr_ds = mydataset.MyDataset_MRI('train', mydataset.train_transforms, args.fold)
tr = DataLoader(tr_ds, args.batchsize, True, pin_memory=True)
ts_ds = mydataset.MyDataset_MRI('test', mydataset.val_transforms, args.fold)
ts = DataLoader(ts_ds, args.batchsize, pin_memory=True)

def test(val):
    model.eval()
    val_avg_acc = 0
    for X, Y in tqdm(val):
        X = X.to(device, dtype=torch.float)
        Y = Y.to(device, torch.int64)
        Yhat = model(X)
        Khat = model.to_classes(model.to_proba(Yhat), 'mode')
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
        for X, Y in tqdm(tr):
            X = X.to(device, dtype=torch.float)
            Y = Y.to(device, torch.int64)
            opt.zero_grad()
            Yhat = model(X)
            loss = model.loss(Yhat, Y)
            loss.backward()
            opt.step()
            Khat = model.to_classes(model.to_proba(Yhat), 'mode')
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
        for X, _ in data:
            phat = model.to_proba(model(X.to(device, dtype=torch.float)))
            Phat += list(phat.cpu().numpy())
    return Phat

proposal = args.method in ('CO', 'CO2', 'HO2')
prefix = '-'.join(f'{k}-{v}' for k, v in vars(args).items())

model = getattr(mymodels, args.method)(args.architecture)

model = model.to(device)
opt = optim.Adam(model.parameters(), args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True)
train(tr, ts)
np.savetxt('output-' + prefix + '-proba.txt', predict_proba(ts), delimiter=',')