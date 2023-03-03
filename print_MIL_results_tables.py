from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, recall_score, precision_score,roc_auc_score, roc_curve, auc, confusion_matrix,balanced_accuracy_score
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from itertools import cycle
import os 
import numpy as np
import pandas as pd
import argparse
from scipy.special import softmax
import torch

parser = argparse.ArgumentParser(description='print results')
parser.add_argument('--fold', type=str, default="all", help='select kfold')
parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size (default: 128)')
parser.add_argument('--s', default=10, type=int, help='how many top k tiles to consider per class(default: 10)')
parser.add_argument('--mix', default='expected', help='get top classes probabilities (3 classes)')
parser.add_argument('--ndims', default=256, type=int, help='length of hidden representation (default: 128)')
parser.add_argument('--weights', default='CE', help='unbalanced positive class weight (default: CE, ordinal)')
parser.add_argument('--results_folder',default='35-Partial_dataset_MIL_bin_MRI_multi_t1ce_flair_fusion_CE_level')

global args
args = parser.parse_args()

def to_proba(Yhat):
        # we need to convert mass distribution into probabilities
        # i.e. P(Y>k) into P(Y=k)
        # P(Y=0) = 1-P(Y>0)
        # P(Y=1) = P(Y>0)-P(Y>1)
        # ...
        # P(Y=K-1) = P(Y>K-2)
        probs = torch.sigmoid(torch.tensor(Yhat))
        probs = torch.cat((1-probs[:, :1], probs[:, :-1]-probs[:, 1:], probs[:, -1:]), 1)
        # there may be small discrepancies
        probs = torch.clamp(probs, 0, 1)
        probs = probs / probs.sum(1, keepdim=True)
        return probs.cpu().numpy()

if args.fold in ("all"):
    folds=[0, 1, 2, 3, 4]
else:
    folds=[0]
    
for fold in folds:

    df= pd.read_csv(os.path.join(args.results_folder,f'predictions_tiles_{args.s}_mix_{args.mix}_loss_{args.weights}_fold_{fold}.csv'))
    
    prefix=f'results_{args.s}_mix_{args.mix}_loss_{args.weights}_fold_{fold}.txt'
    
    Y_true=df.loc[:,'target'].tolist()
    Phat=df.loc[:,'prediction'].tolist()
    Phat_0=df.loc[:,'probability_0'].tolist()
    Phat_1=df.loc[:,'probability_1'].tolist()
    Phat_2=df.loc[:,'probability_2'].tolist()
    
    if args.weights in ("ordinal"):
        Phat_p = to_proba([(Phat_0[i],Phat_1[i],Phat_2[i]) for i in range(len(Phat_0))])[:,0:3]
        Phat_p = softmax(Phat_p, axis=1)
    else: 
        Phat_p=np.array([softmax([Phat_0[i],Phat_1[i],Phat_2[i]]) for i in range(len(Phat_0))])
    
    
    
    
    accuracy_balanced = balanced_accuracy_score(Y_true, Phat)
    accuracy = accuracy_score(Y_true, Phat)
    mae = mean_absolute_error(Y_true, Phat)
    f1 = f1_score(Y_true, Phat, average='weighted')
    recall = recall_score(Y_true, Phat, average='weighted')
    precision = precision_score(Y_true, Phat, average='weighted')
    # aucc = roc_auc_score(Y_true,Phat_p, multi_class='ovr')
    try:
      aucc = roc_auc_score(Y_true,Phat_p, multi_class='ovr')
    except:
      aucc = 0
    
    
    
    cm = confusion_matrix(Y_true, Phat, normalize="true")
    
    cm_n = confusion_matrix(Y_true, Phat, normalize=None)
    
    os.makedirs(os.path.join(args.results_folder,"results"), exist_ok=True)
    
    f = open(os.path.join(args.results_folder,"results", (prefix)), 'a+')
    f.write('\n\nModel:'+str(prefix)+
        ' \naccuracy:'+ str(accuracy)+
        ' \nmae:'+ str(mae)+
        ' \nf1:'+str(f1)+
        ' \nrecall:'+ str(recall)+
        ' \nprecision:'+ str(precision)+
        ' \nauc:'+ str(aucc)+
        ' \nbalanced_accuracy:'+ str(accuracy_balanced))
    f.close()
    
    import pandas as pd
    import seaborn as sn
    import matplotlib.pyplot as plt
    
    
    classes = ["0", "1", "2"]
    
    df_cfm = pd.DataFrame(cm, index = classes, columns = classes)
    plt.figure(figsize = (10,7))
    plt.rcParams.update({'font.size': 56})
    cfm_plot = sn.heatmap(df_cfm, annot=True,cmap="BuPu", cbar=False)
    cfm_plot.figure.savefig(os.path.join(args.results_folder,"results", (os.path.splitext(prefix))[0])+"_cfm_normalized.png",dpi=200)
    
    df_cfm = pd.DataFrame(cm_n, index = classes, columns = classes)
    plt.figure(figsize = (10,7))
    plt.rcParams.update({'font.size': 56})
    cfm_plot = sn.heatmap(df_cfm, annot=True,cmap="BuPu", cbar=False)
    cfm_plot.figure.savefig(os.path.join(args.results_folder,"results", (os.path.splitext(prefix))[0])+"_cfm.png",dpi=200)