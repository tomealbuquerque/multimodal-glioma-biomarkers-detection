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
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
from order_top_data import create_max_mix_tiles_dataset, create_max_tiles_dataset, create_max_tiles_dataset_bin,create_max_mix_tiles_dataset_bin,create_max_tiles_dataset_expected_value_bin
from torch.utils.data import Dataset
from torchvision import models, transforms
import monai
from monai.data import ImageDataset, DataLoader
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity


parser = argparse.ArgumentParser(description='Multimodal MRI + Path - MLP aggregator - context aware')
parser.add_argument('--train_lib', type=str, default='train', help='path to train MIL library binary')
parser.add_argument('--val_lib', type=str, default='test', help='path to validation MIL library binary. If present.')
parser.add_argument('--fold', type=int, default=4, help='select kfold')
parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size (default: 128)')
parser.add_argument('--nepochs', type=int, default=20, help='number of epochs')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--s', default=5, type=int, help='how many top k tiles to consider per class(default: 10)')
parser.add_argument('--mix', default='global',choices=['mix','global', 'expected'], help='get top classes probabilities (3 classes)')
parser.add_argument('--ndims', default=256, type=int, help='length of hidden representation (default: 128)')
parser.add_argument('--model_PATH', type=str, default='checkpoint_best_512_bin_fold_4.pth', help='path to trained model checkpoint')
parser.add_argument('--model_PATH_level', type=str, default='checkpoint_best_512_bin_multi_level_fold_4.pth', help='path to trained model checkpoint')
parser.add_argument('--model_MRI', type=str, default='MRI_weights\multiclass_t1ce_flair\multiclass_fold4_t1ce_flair\multiclass_checkpoint_best_tiles.pth', help='path to trained model checkpoint')
parser.add_argument('--method_MRI', choices=['UniMRI','MultiMRI'], default='UniMRI')
parser.add_argument('--MRI_type', choices=['flair', 't1', 't1ce', 't2', 'all'], default='t1ce')
parser.add_argument('--MRI_n_outputs', default=3, type=int,help='multiclass vs binary mri')
parser.add_argument('--weights', default='CE', help='unbalanced positive class weight (default: CE, ordinal)')
parser.add_argument('--shuffle', action='store_true', help='to shuffle order of sequence')
parser.add_argument('--dataset', default='',choices=['','_full', '_all','_biomarkers_all','_biomarkers_full'], help='select dataset partition')
parser.add_argument('--results_folder',default='teste')

def main():
    
    global args, best_loss
    args = parser.parse_args()
    best_loss=0
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
    
    #MRI transforms
    train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), RandRotate90()])
    val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst()])
    
    train_dset = rnndata(fold=args.fold, typet=args.train_lib, s=args.s, shuffle=args.shuffle, transform=trans, transform_MRI=val_transforms, MRI_exam=args.MRI_type)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=args.shuffle,
        num_workers=args.workers, pin_memory=False)
    
    
    val_dset = rnndata(fold=args.fold, typet=args.val_lib, s=args.s, shuffle=False, transform=trans,transform_MRI=val_transforms, MRI_exam=args.MRI_type)
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    #make model fro PATH
    embedder_PATH = ResNetEncoder(args.model_PATH)
    for param in embedder_PATH.parameters():
        param.requires_grad = False
    embedder_PATH = embedder_PATH.cuda()
    embedder_PATH.eval()
    
    
    embedder_PATH_level_2 = ResNetEncoder_level(args.model_PATH_level)
    for param in embedder_PATH_level_2.parameters():
        param.requires_grad = False
    embedder_PATH_level_2 = embedder_PATH_level_2.cuda()
    embedder_PATH_level_2.eval()
    
    
    #make model for MRI
    embedder_MRI = MRI_DensenetEncoder(args.model_MRI)
    for param in embedder_MRI.parameters():
        param.requires_grad = False
    embedder_MRI = embedder_MRI.cuda()
    embedder_MRI.eval()

    mlp = MLP_single(args.ndims)
    # mlp_dict = torch.load(os.path.join(args.results_folder,f'rnn_checkpoint_best_tiles_{args.s}_mix_{args.mix}_loss_{args.weights}_fold_{args.fold}.pth'))
    # mlp.load_state_dict(mlp_dict['state_dict'])
    mlp = mlp.cuda()
    

    if args.weights in ('CE'):
        # w = torch.Tensor([0.5912698412698413, 1.1037037037037036, 2.4833333333333334])
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion=ordinal_loss
    
    #create results_folder
    os.makedirs(os.path.join(args.results_folder,"results"), exist_ok=True)

    optimizer = optim.Adam(mlp.parameters(), lr=1e-3)
    cudnn.benchmark = True

    fconv = open(os.path.join(args.results_folder, f'convergence_tiles_{args.s}_mix_{args.mix}_loss_{args.weights}_fold_{args.fold}.csv'), 'w')
    fconv.write('epoch,train.loss,val.loss\n')
    fconv.close()

    
    for epoch in range(args.nepochs):

        train_loss = train_single(epoch, embedder_PATH,embedder_PATH_level_2,embedder_MRI, mlp, train_loader, criterion, optimizer)
        val_loss, avg_acc, Phat, probs = test_single(epoch, embedder_PATH,embedder_PATH_level_2,embedder_MRI, mlp, val_loader, criterion)

        fconv = open(os.path.join(args.results_folder,f'convergence_tiles_{args.s}_mix_{args.mix}_loss_{args.weights}_fold_{args.fold}.csv'), 'a')
        fconv.write('{},{},{}\n'.format(epoch+1, train_loss, val_loss))
        fconv.close()

        val_err = avg_acc
        if val_err > best_loss:
            best_loss = val_err
            obj = {
                'epoch': epoch+1,
                'state_dict': mlp.state_dict()
            }
            torch.save(obj, os.path.join(args.results_folder,f'rnn_checkpoint_best_tiles_{args.s}_mix_{args.mix}_loss_{args.weights}_fold_{args.fold}.pth'))
             
          
            fp = open(os.path.join(args.results_folder, f'predictions_tiles_{args.s}_mix_{args.mix}_loss_{args.weights}_fold_{args.fold}.csv'), 'w')
            fp.write('file,target,prediction,probability_0,probability_1,probability_2\n')
            for name, target, prob, pred in zip(val_dset.slidenames, val_dset.targets, probs,Phat):
                
                fp.write('{},{},{},{},{},{}\n'.format(name, target, pred, prob[0],prob[1],prob[2]))
                
            fp.close()
            
def train_single(epoch, embedder_PATH,embedder_PATH_level_2, embedder_MRI, mlp, loader, criterion, optimizer):
    mlp.train()
    running_loss = 0.
    avg_acc = 0
    avg_auc=0
    avg_mae=0
    Phat=[]
    Phat_p=[]
    Y_true=[]

    for i,(inputs_PATH,inputs_PATH_level,inputs_MRI,inputs_age, inputs_gender, target) in enumerate(loader):
        print('Training - Epoch: [{}/{}]\tBatch: [{}/{}]'.format(epoch+1, args.nepochs, i+1, len(loader)))

        batch_size = inputs_PATH[0].size(0)
        mlp.zero_grad()

        inputs=[]
        inputs_level=[]


        for s in range(len(inputs_PATH)):
            input1 = inputs_PATH[s].cuda()
            input2 = inputs_PATH_level[s].cuda()
            _, input1 = embedder_PATH(input1)
            _, input2 = embedder_PATH_level_2(input2)
            inputs.append(input1)
            inputs_level.append(input2)
        
        input_mri, classes_mri = embedder_MRI(inputs_MRI.cuda())
        # input = torch.cat((input_mri, torch.cat(inputs, dim=1),torch.unsqueeze(inputs_age.cuda(), 1).float(),torch.unsqueeze(inputs_gender.cuda(), 1).float()), dim=1)
        input = torch.cat((input_mri, torch.cat(inputs, dim=1),torch.cat(inputs_level, dim=1)), dim=1)
        output = mlp(input)

    
        target = target.cuda()
        loss = criterion(output,target)

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()*target.size(0)
         
        probss, classes = to_proba_and_classes(output)
        GT = target.detach().cpu().numpy()
        if args.weights in ('ordinal'):
            Pred=(classes.detach().squeeze().cpu().numpy())
        else:
            Pred=(output.detach().squeeze().cpu().numpy().argmax(1))*1
            
        Phat_p.append(F.softmax(output, 1).detach().cpu().numpy())

        Phat.append(Pred)
        Y_true.append(GT)
        
    Phat = np.concatenate(Phat)
    Y_true = np.concatenate(Y_true)
    Phat_p = np.concatenate(Phat_p)
    
    running_loss = running_loss/len(loader.dataset)
    avg_acc = accuracy_score(Y_true, Phat)
    avg_mae = mean_absolute_error(Y_true, Phat)
    # avg_auc = roc_auc_score(Y_true,Phat_p)

    print('Training - Epoch: [{}/{}]\tLoss: {}\tACC: {}\tMAE: {}'.format(epoch+1, args.nepochs, running_loss, avg_acc, avg_mae))
    return running_loss

def test_single(epoch,embedder_PATH,embedder_PATH_level_2, embedder_MRI, mlp, loader, criterion):
    mlp.eval()
    running_loss = 0.
    avg_acc = 0
    avg_auc=0
    avg_mae=0
    Phat=[]
    Y_true=[]
    Phat_p=[]
    
    with torch.no_grad():
        for i,(inputs_PATH,inputs_PATH_level,inputs_MRI,inputs_age, inputs_gender, target)  in enumerate(loader):
            print('Validating - Epoch: [{}/{}]\tBatch: [{}/{}]'.format(epoch+1,args.nepochs,i+1,len(loader)))
            
            batch_size = inputs_PATH[0].size(0)
            inputs=[]
            inputs_level=[]
    
    
            for s in range(len(inputs_PATH)):
                input1 = inputs_PATH[s].cuda()
                input2 = inputs_PATH_level[s].cuda()
                _, input1 = embedder_PATH(input1)
                _, input2 = embedder_PATH_level_2(input2)
                inputs.append(input1)
                inputs_level.append(input2)
            
            input_mri, classes_mri = embedder_MRI(inputs_MRI.cuda())
            # input = torch.cat((input_mri, torch.cat(inputs, dim=1),torch.unsqueeze(inputs_age.cuda(), 1).float(),torch.unsqueeze(inputs_gender.cuda(), 1).float()), dim=1)
            input = torch.cat((input_mri, torch.cat(inputs, dim=1),torch.cat(inputs_level, dim=1)), dim=1)
            output = mlp(input)
            

            target = target.cuda()
            loss = criterion(output,target)
            
            running_loss += loss.item()*target.size(0)
            
            probss, classes = to_proba_and_classes(output)
            GT = target.detach().cpu().numpy()
            if args.weights in ('ordinal'):
                Pred=(classes.detach().squeeze().cpu().numpy())
            else:
                Pred=(output.detach().squeeze().cpu().numpy().argmax(1))*1
                
            Phat_p.append(F.softmax(output, 1).detach().cpu().numpy())
            Phat.append(Pred)
            Y_true.append(GT)
            
 
        Phat = np.concatenate(Phat)
        Y_true = np.concatenate(Y_true)
        probs = np.concatenate(Phat_p)
        

        print(Y_true)
        print(Phat)
        running_loss = running_loss/len(loader.dataset)
        avg_acc = accuracy_score(Y_true, Phat)
        avg_mae = mean_absolute_error(Y_true, Phat)

    
    print('Validating - Epoch: [{}/{}]\tLoss: {}\tACC: {}\tMAE: {}'.format(epoch+1, args.nepochs, running_loss, avg_acc,avg_mae))
    return running_loss,avg_acc,Phat, probs



class MRI_DensenetEncoder(nn.Module):

    def __init__(self, path):
        super(MRI_DensenetEncoder, self).__init__()
        # path='trial_2023_01_05_21_54_42/best_metric_model_classification3d_array.pth'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # temp = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=args.MRI_n_outputs, dropout_prob=0.15)
        # temp.load_state_dict(torch.load(path))
        # temp.load_state_dict(torch.load(path)['state_dict'])
        temp1 = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=args.MRI_n_outputs, dropout_prob=0.15)
        temp1.load_state_dict(torch.load(path)['state_dict'])
        # temp.to(device)
        temp1.to(device)
        temp1.class_layers.out = nn.Linear(temp1.class_layers.out.in_features, 1024)
        
        
        # temp = temp.model[0:7]
        
        self.temp1= temp1
        # self.temp= temp
      
    def forward(self,x):
        x1 = self.temp1(x)
        # x2 = self.temp(x)
        return x1, x1

class ResNetEncoder(nn.Module):

    def __init__(self, path):
        super(ResNetEncoder, self).__init__()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        temp = models.resnet34(True)
        temp.fc = nn.Linear(temp.fc.in_features, 2)
        ch = torch.load(args.model_PATH)
        temp= nn.DataParallel(temp)
        temp.load_state_dict(ch['state_dict'])
        temp.to(device)
        
        self.features = nn.Sequential(*list(temp.module.children())[:-1])
        self.fc = temp.module.fc

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        return self.fc(x), x
    

class ResNetEncoder_level(nn.Module):

    def __init__(self, path_level):
        super(ResNetEncoder_level, self).__init__()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        temp = models.resnet34(True)
        temp.fc = nn.Linear(temp.fc.in_features, 2)
        ch = torch.load(path_level)
        temp= nn.DataParallel(temp)
        temp.load_state_dict(ch['state_dict'])
        temp.to(device)
        
        self.features = nn.Sequential(*list(temp.module.children())[:-1])
        self.fc = temp.module.fc

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        return self.fc(x), x
    
    
class MLP_single(nn.Module):

    def __init__(self, ndims):
        super(MLP_single, self).__init__()
        self.ndims = ndims
        
        self.model = nn.Sequential(
    # +args.MRI_n_outputs+2
            nn.Linear(args.s*1024+1024, 512),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )


    def forward(self, input):
        output = self.model(input)
        return output

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.ndims)



class rnndata(data.Dataset):

    def __init__(self, fold, typet, s, shuffle=False, transform=None, transform_MRI=None, MRI_exam='flair'):
        
        
        lib, targets = pickle.load(open(f'multimodal_glioma_data_sorted{args.dataset}.pickle', 'rb'))[0][typet]
        
        
        
        slide_names=[lib[i]['slide'] for i in range(len(lib))]
        target=[[targets[i]['idh1'], targets[i]['ioh1p15q']] for i in range(len(targets))]
        pat_age=[lib[i]['age'] for i in range(len(lib))]
        pat_gender=[lib[i]['gender'] for i in range(len(lib))]
                
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
        self.age =pat_age
        self.gender = pat_gender
        self.transform = transform
        self.mode = None
        self.mult = 0.5
        self.size = int(np.round(2048*1))
        self.level = 0
        self.s = s
        self.shuffle = shuffle
        
        #MRI
        self.transform_MRI = transform_MRI
        self.MRI_exam=MRI_exam
        self.lib=lib

    def __getitem__(self,index):
        
        #MRI
        if self.MRI_exam ==args.MRI_type:
            img_flair = np.load(os.path.join('data_multimodal_tcga',self.lib[index][args.MRI_type+'_block']))
            X_flair = self.transform_MRI(img_flair)
            mri=X_flair
        
        #PATHOLOGY
        slide = self.slides[index]
        grid = self.grid[index]
        if self.shuffle:
            grid = random.sample(grid,len(grid))

        patho = []
        patho_level = []
        s = min(self.s, len(grid))
        for i in range(s):
            img_level = slide.read_region((grid[i][0]-768,grid[i][1]-768), self.level, (self.size, self.size)).convert('RGB')
            img = slide.read_region(grid[i], self.level, (512, 512)).convert('RGB')
            if self.mult != 1:
                img_level = img_level.resize((512,512), Image.BILINEAR)
            if self.transform is not None:
                img_level = self.transform(img_level)
                img = self.transform(img)
            patho.append(img)
            patho_level.append(img_level)
        return patho,patho_level, mri, self.age[index], self.gender[index], self.targets[index]

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
