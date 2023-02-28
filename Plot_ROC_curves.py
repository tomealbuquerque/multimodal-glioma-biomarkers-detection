# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:53:36 2023

@author: albu
"""
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
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, recall_score, precision_score,roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from itertools import cycle


parser = argparse.ArgumentParser(description='print results')
parser.add_argument('--fold', type=str, default="all", help='select kfold')
parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size (default: 128)')
parser.add_argument('--s', default=10, type=int, help='how many top k tiles to consider per class(default: 10)')
parser.add_argument('--mix', default='expected', help='get top classes probabilities (3 classes)')
parser.add_argument('--ndims', default=256, type=int, help='length of hidden representation (default: 128)')
parser.add_argument('--weights', default='CE', help='unbalanced positive class weight (default: CE, ordinal)')
parser.add_argument('--results_folder',default='29-Partial_dataset_MIL_bin_MRI_multi_t1ce_flair_late_fusion_CE')


global args
args = parser.parse_args()



def get_AUC_plot(results_folder,model_type,mix,name,s=10):
    
    if args.fold in ("all"):
        folds=[0, 1, 2, 3, 4]
    else:
        folds=[0]
    
    Phat_p_all=[]
    Y_true_all=[]
    Phat_all=[]
    for fold in folds:
        
        if model_type=="MRI":
            
            df= pd.read_csv(os.path.join(results_folder,f'multiclass_fold{fold}_t1ce_on_t1ce_block.csv'))
        else:
            
            df= pd.read_csv(os.path.join(results_folder,f'predictions_tiles_{s}_mix_{mix}_loss_{args.weights}_fold_{fold}.csv'))
        
        
        Y_true=df.loc[:,'target'].tolist()
        Phat=df.loc[:,'prediction'].tolist()
        Phat_0=df.loc[:,'probability_0'].tolist()
        Phat_1=df.loc[:,'probability_1'].tolist()
        Phat_2=df.loc[:,'probability_2'].tolist()
        
    
        Phat_p=np.array([softmax([Phat_0[i],Phat_1[i],Phat_2[i]]) for i in range(len(Phat_0))])
        
        Phat_p_all.append(Phat_p)
        
        Y_true_all.append(label_binarize(Y_true, classes=[0, 1, 2]))
        Phat_all.append(label_binarize(Phat, classes=[0, 1, 2]))
    
    y_test = Y_true_all
    n_classes = Y_true_all[0].shape[1]
    Phat = Phat_all
    y_score = np.array(Phat_p_all)
    
    
    fig = plt.figure(1)
    
    fpr_all=[]
    tpr_all=[]
    roc_auc_all=[]
    mean_fpr=[]
    
    mean_fpr = np.linspace(0, 1, 100)
    tprs_all=[]
    for fold in folds:
    
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        tprs = []
        aucs=[]
        for i in range(n_classes):
            # print(Phat)
            fpr[i], tpr[i], _ = roc_curve(y_test[fold][:, i], y_score[fold][:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
            
            tprs.append(np.interp(mean_fpr, fpr[i], tpr[i]))
            tprs[-1][0] = 0.0
            aucs.append(np.array(roc_auc[i]))
            
            
        # colors= [plt.cm.Pastel1(i) for i in range(20)]
        
        if model_type=='H&E':
            color=plt.cm.tab10(6)
        elif model_type=='MRI':
            color=plt.cm.tab10(7)
        else :
            color=plt.cm.tab10(0)
            
        roc_auc_all.append(aucs)
        tprs_all.append(np.mean(np.array(tprs),axis=0))  
        mean_fpr = np.linspace(0, 1, 100)   
        mean_tpr = np.mean(tprs_all, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        # print(mean_auc)
        std_auc = np.std(roc_auc_all)
        print(mean_auc,'+-',std_auc)
        plt.plot(mean_fpr, mean_tpr, color=color,
                 # label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 label=name,
                 lw=2, alpha=1)

        std_tpr = np.std(tprs_all, axis=1)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=.15,
                         label=r'')
        # label=r'$\pm$ std. dev.')
    
            
        return roc_auc_all,tprs_all
    
# roc_auc_all,tprs_all=get_AUC_plot(results_folder='28-Partial_dataset_MIL_bin_multiclass_kfold',model_type="H&E",mix="global")  
roc_auc_all,tprs_all=get_AUC_plot(results_folder='28-Partial_dataset_MIL_bin_multiclass_kfold',model_type="H&E",mix="expected",name="H&E - Expected tiles    (AUC = 0.815 ± 0.088)")    
roc_auc_all,tprs_all=get_AUC_plot(results_folder='results_MRI',model_type="MRI",mix="global",name="MRI - T1ce Segment.    (AUC = 0.863 ± 0.016)")    
roc_auc_all,tprs_all=get_AUC_plot(results_folder='31-Partial_dataset_MIL_bin_MRI_multi_t1ce_flair_fusion_CE',model_type="Multimodal",mix="expected",name="Multimodal Mid Fusion (AUC = 0.875 ± 0.062)") 
# roc_auc_all,tprs_all=get_AUC_plot(results_folder='41-Partial_dataset_MIL_bin_MRI_multi_t1ce_flair_fusion_CE_level_over',model_type="Multimodal",mix="expected",name="Multimodal Context (AUC = 0.875 ± 0.062)",s=5) 

    
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray',
         label='', alpha=1)
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate',fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate',fontsize=12, fontweight='bold')
# plt.title('ROC for 5-fold models',fontsize=14,fontweight='bold')
plt.legend(loc="lower right", prop={'size': 8.8},title='Models',fontsize=14)
# plt.grid(color='gray', linestyle='-', linewidth=1, alpha=0.1)
plt.savefig('ROC_AUC_plot.png', bbox_inches='tight', dpi=200)
plt.show()

