import os
import sys
import openslide
from PIL import Image
import numpy as np
import random
import argparse
import pickle
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
from order_top_data import create_max_mix_tiles_dataset, create_max_tiles_dataset, create_max_tiles_dataset_bin,create_max_mix_tiles_dataset_bin,create_max_tiles_dataset_expected_value_bin

parser = argparse.ArgumentParser(description='MIL model with aggregator on top training script')
parser.add_argument('--train_lib', type=str, default='train', help='path to train MIL library binary')
parser.add_argument('--val_lib', type=str, default='test', help='path to validation MIL library binary. If present.')
parser.add_argument('--fold', type=int, default=0, help='select kfold')
parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size (default: 128)')
parser.add_argument('--nepochs', type=int, default=30, help='number of epochs')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--s', default=4, type=int, help='how many top k tiles to consider per class(default: 10)')
parser.add_argument('--mix', default='global',choices=['mix','global', 'expected'], help='get top classes probabilities (3 classes)')
parser.add_argument('--ndims', default=256, type=int, help='length of hidden representation (default: 128)')
parser.add_argument('--model', type=str, default='checkpoint_best_512_bin_fold_0.pth', help='path to trained model checkpoint')
parser.add_argument('--weights', default='ordinal', help='unbalanced positive class weight (default: CE, ordinal)')
parser.add_argument('--shuffle', action='store_true', help='to shuffle order of sequence')
parser.add_argument('--dataset', default='',choices=['','_full', '_all','_biomarkers_all','_biomarkers_full'], help='select dataset partition')
parser.add_argument('--results_folder',default='')


def main():
    best_loss=0
    global args, best_acc
    args = parser.parse_args()
    
    #Create dataset with top tiles
    if args.mix in ('mix'):
        print(f'Top tiles per class: {round(args.s/2)}')
        create_max_mix_tiles_dataset_bin('multimodal_glioma_data_sorted{args.dataset}.pickle',args.fold, round(args.s/2),args.dataset)
    elif args.mix in ('expected'): 
        print(f'Top tiles: {args.s}')
        create_max_tiles_dataset_expected_value_bin('multimodal_glioma_data_sorted{args.dataset}.pickle',args.fold,args.dataset)
    else: 
        print(f'Top tiles: {args.s}')
        create_max_tiles_dataset_bin('multimodal_glioma_data_sorted{args.dataset}.pickle',args.fold, args.s,args.dataset)
        
    #load libraries
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    train_dset = rnndata(fold=args.fold, typet=args.train_lib,dataset=args.dataset, s=args.s, shuffle=args.shuffle, transform=trans)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=args.shuffle,
        num_workers=args.workers, pin_memory=False)
    val_dset = rnndata(fold=args.fold, typet=args.val_lib,dataset=args.dataset, s=args.s, shuffle=False, transform=trans)
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    #make model
    embedder = ResNetEncoder(args.model)
    for param in embedder.parameters():
        param.requires_grad = False
    embedder = embedder.cuda()
    embedder.eval()

    rnn = rnn_single(args.ndims)
    # rnn_dict = torch.load(os.path.join(args.results_folder,f'rnn_checkpoint_best_tiles_{args.s}_mix_{args.mix}_loss_{args.weights}.pth'))
    # rnn.load_state_dict(rnn_dict['state_dict'])
    rnn = rnn.cuda()
    

    if args.weights in ('CE'):
        # w = torch.Tensor([0.5912698412698413, 1.1037037037037036, 2.4833333333333334])
        criterion = nn.CrossEntropyLoss().cuda()
    elif args.weights in ('bin'):
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion=ordinal_loss
        
    #create results_folder
    os.makedirs(os.path.join(args.results_folder,"results"), exist_ok=True)
    # optimizer = optim.SGD(rnn.parameters(), 0.1, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True)
    optimizer = optim.Adam(rnn.parameters(), lr=1e-4)
    cudnn.benchmark = True

    fconv = open(os.path.join(args.results_folder, f'convergence_tiles_{args.s}_mix_{args.mix}_loss_{args.weights}_fold_{args.fold}{args.dataset}.csv'), 'w')
    fconv.write('epoch,train.loss,val.loss\n')
    fconv.close()

    
    for epoch in range(args.nepochs):

        train_loss = train_single(epoch, embedder, rnn, train_loader, criterion, optimizer)
        val_loss, avg_acc, Phat, probs = test_single(epoch, embedder, rnn, val_loader, criterion)

        fconv = open(os.path.join(args.results_folder,f'convergence_tiles_{args.s}_mix_{args.mix}_loss_{args.weights}_fold_{args.fold}{args.dataset}.csv'), 'a')
        fconv.write('{},{},{}\n'.format(epoch+1, train_loss, val_loss))
        fconv.close()

        val_err = avg_acc
        if val_err > best_loss:
            best_loss = val_err
            obj = {
                'epoch': epoch+1,
                'state_dict': rnn.state_dict()
            }
            torch.save(obj, os.path.join(args.results_folder,f'rnn_checkpoint_best_tiles_{args.s}_mix_{args.mix}_loss_{args.weights}_fold_{args.fold}{args.dataset}.pth'))

            fp = open(os.path.join(args.results_folder, f'predictions_tiles_{args.s}_mix_{args.mix}_loss_{args.weights}_fold_{args.fold}.csv'), 'w')
            fp.write('file,target,prediction,probability_0,probability_1,probability_2\n')
            for name, target, prob, pred in zip(val_dset.slidenames, val_dset.targets, probs,Phat):
                
                fp.write('{},{},{},{},{},{}\n'.format(name, target, pred, prob[0],prob[1],prob[2]))
                
            fp.close()



def train_single(epoch, embedder, rnn, loader, criterion, optimizer):
    rnn.train()
    running_loss = 0.
    avg_acc = 0
    avg_auc=0
    avg_mae=0
    Phat=[]
    Phat_p=[]
    Y_true=[]
    for i,(inputs,target) in enumerate(loader):
        print('Training - Epoch: [{}/{}]\tBatch: [{}/{}]'.format(epoch+1, args.nepochs, i+1, len(loader)))

        batch_size = inputs[0].size(0)
        rnn.zero_grad()

        state = rnn.init_hidden(batch_size).cuda()
        for s in range(len(inputs)):
            input = inputs[s].cuda()
            _, input = embedder(input)
            output, state = rnn(input, state)

        target = target.cuda()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()*target.size(0)

         
        probs, classes = to_proba_and_classes(output)
        GT = target.detach().cpu().numpy()
        if args.weights in ('ordinal'):
            Pred=(classes.detach().squeeze().cpu().numpy())
        else:
            Pred=(output.detach().squeeze().cpu().numpy().argmax(1))*1
        # print('GT: ', GT,' \nPred: ', Pred)
        Phat.append(Pred)
        Y_true.append(GT)
    
    # Phat_p = np.concatenate(Phat_p)
    Phat = np.concatenate(Phat)
    Y_true = np.concatenate(Y_true)
    
    running_loss = running_loss/len(loader.dataset)
    avg_acc = accuracy_score(Y_true, Phat)
    avg_mae = mean_absolute_error(Y_true, Phat)
    # avg_auc = roc_auc_score(Y_true,Phat_p, multi_class='ovr')

    print('Training - Epoch: [{}/{}]\tLoss: {}\tACC: {}\tMAE: {}'.format(epoch+1, args.nepochs, running_loss, avg_acc, avg_mae))
    return running_loss

def test_single(epoch, embedder, rnn, loader, criterion):
    rnn.eval()
    running_loss = 0.
    avg_acc = 0
    avg_auc=0
    avg_mae=0
    Phat=[]
    Y_true=[]
    Phat_p=[]
    
    with torch.no_grad():
        for i,(inputs,target) in enumerate(loader):
            print('Validating - Epoch: [{}/{}]\tBatch: [{}/{}]'.format(epoch+1,args.nepochs,i+1,len(loader)))
            
            batch_size = inputs[0].size(0)
            
            state = rnn.init_hidden(batch_size).cuda()
            for s in range(len(inputs)):
                input = inputs[s].cuda()
                _, input = embedder(input)
                output, state = rnn(input, state)

            target = target.cuda()
            loss = criterion(output,target)
            
            running_loss += loss.item()*target.size(0)

            probs, classes = to_proba_and_classes(output)
            GT = target.detach().cpu().numpy()
            if args.weights in ('ordinal'):
                Pred=(classes.detach().squeeze().cpu().numpy())
                Phat_p.append(probs.detach().cpu().numpy())
            else:
                Pred=(output.detach().squeeze().cpu().numpy().argmax(1))*1
                Phat_p.append(F.softmax(output, 1).detach().cpu().numpy())
            # print('GT: ', GT,' \nPred: ', Pred)
            Phat.append(Pred)
            Y_true.append(GT)
           
            
        probs = np.concatenate(Phat_p)
        # Phat_p = np.concatenate(Phat_p)
        Phat = np.concatenate(Phat)
        Y_true = np.concatenate(Y_true)
        
        running_loss = running_loss/len(loader.dataset)
        avg_acc = accuracy_score(Y_true, Phat)
        avg_mae = mean_absolute_error(Y_true, Phat)
        # avg_auc = roc_auc_score(Y_true,Phat_p, multi_class='ovr')

    
    print('Validating - Epoch: [{}/{}]\tLoss: {}\tACC: {}\tMAE: {}'.format(epoch+1, args.nepochs, running_loss, avg_acc, avg_mae))
    return running_loss,avg_acc, Phat, probs


class ResNetEncoder(nn.Module):

    def __init__(self, path):
        super(ResNetEncoder, self).__init__()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        temp = models.resnet34(True)
        temp.fc = nn.Linear(temp.fc.in_features, 2)
        ch = torch.load(args.model)
        temp= nn.DataParallel(temp)
        temp.load_state_dict(ch['state_dict'])
        temp.to(device)
        
        
        
        self.features = nn.Sequential(*list(temp.module.children())[:-1])
        self.fc = temp.module.fc

    def forward(self,x):
        x = self.features(x)

        x = x.view(x.size(0),-1)
        return self.fc(x), x

class rnn_single(nn.Module):

    def __init__(self, ndims):
        super(rnn_single, self).__init__()
        self.ndims = ndims
        
        self.model = nn.Sequential(
         nn.Linear(512, ndims),
         nn.Dropout(0.1),
         nn.ReLU(),
         nn.Linear(ndims, ndims),
         nn.ReLU())
        
        self.fc4 = nn.Linear(ndims, ndims)
        self.fc5 = nn.Linear(ndims, 3)
        self.activation = nn.ReLU()

    def forward(self, input, state):
        input = self.model(input)
        state = self.fc4(input)
        state = self.activation(state+input)
        output = self.fc5(state)
        return output, state

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.ndims)

class rnndata(data.Dataset):

    def __init__(self, fold, typet,dataset, s, shuffle=False, transform=None):
        
        
        lib, targets = pickle.load(open(f'multimodal_glioma_data_sorted{dataset}.pickle', 'rb'))[0][typet]
        
        
        
        if dataset =='_full':

            slide_names=[lib[i]['slide'] for i in range(len(lib))]
            target=[[targets[i]['idh1'], targets[i]['ioh1p19q']] for i in range(len(targets))]

            
        else:
                
            slide_names=[lib[i]['slide'] for i in range(len(lib))]
            target=[[targets[i]['idh1'], targets[i]['ioh1p15q']] for i in range(len(targets))]
            
                    
                #Encoding targets 
        for i in range(len(target)):
            if target[i]==[0,0]:
                target[i]=0
            elif target[i]==[1,0]:
                target[i]=1
            else:
                target[i]=2
                
        
        grids=np.array([lib[i]['sorted_coords'] for i in range(len(lib))],dtype=object)
        
        slides = []
        for i,name in enumerate(slide_names):
            slides.append(openslide.OpenSlide(os.path.join(f'E:/Pathology',name)))

        print('Number of slides: {}'.format(len(grids)))
        self.slidenames = slide_names
        self.slides = slides
        self.targets = target
        self.grid = grids
        self.transform = transform
        self.mode = None
        self.mult = 1
        self.size = int(np.round(512*1))
        self.level = 0
        self.s = s
        self.shuffle = shuffle


    def __getitem__(self,index):

        slide = self.slides[index]
        grid = self.grid[index]
        if self.shuffle:
            grid = random.sample(grid,len(grid))

        out = []
        s = min(self.s, len(grid))
        for i in range(s):
            # print(grid[i])
            img = slide.read_region(grid[i], self.level, (self.size, self.size)).convert('RGB')
            if self.mult != 1:
                img = img.resize((224,224), Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            out.append(img)
        
        return out, self.targets[index]

    def __len__(self):
        
        return len(self.targets)
    
#optimization
def ordinal_loss(Yhat, Y):
# if K=3, then
#     Y=0 => Y_=[1, 0, 0]
#     Y=1 => Y_=[1, 1, 0]
#     Y=2 => Y_=[1, 1, 1]
    KK = torch.arange(3, device='cuda').expand(Y.shape[0], -1)
    YY = (Y[:, None] > KK).float()
    return F.binary_cross_entropy_with_logits(Yhat, YY).cuda()
    
def to_proba(Yhat):
        # we need to convert mass distribution into probabilities
        # i.e. P(Y>k) into P(Y=k)
        # P(Y=0) = 1-P(Y>0)
        # P(Y=1) = P(Y>0)-P(Y>1)
        # ...
        # P(Y=K-1) = P(Y>K-2)
        probs = torch.sigmoid(Yhat)
        probs = torch.cat((1-probs[:, :1], probs[:, :-1]-probs[:, 1:], probs[:, -1:]), 1)
        # there may be small discrepancies
        probs = torch.clamp(probs, 0, 1)
        probs = probs / probs.sum(1, keepdim=True)
        return probs

def to_proba_and_classes(ypred):
    probs = to_proba(ypred)
    classes = torch.sum(ypred >= 0, 1)
    return probs, classes

if __name__ == '__main__':
    main()
