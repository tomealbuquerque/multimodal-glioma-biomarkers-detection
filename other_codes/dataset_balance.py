# -*- coding: utf-8 -*-
"""
Get dataset balance
"""

import pickle
import numpy as np
import os

X, Y = pickle.load(open(os.path.join('data_multimodal_tcga',f'multimodal_glioma_data.pickle'), 'rb'))[0]["train"]

Y_n=[]
for i in range(len(Y)):
    Y_n.append([Y[i]['idh1'], Y[i]['ioh1p15q']])
    #Encoding targets 
    if Y_n[i]==[0,0]:
        Y_n[i]=0
    elif Y_n[i]==[1,0]:
        Y_n[i]=1
    else:
        Y_n[i]=2
        
from collections import Counter


print(Counter(Y_n).keys()) # equals to list(set(words))
print(Counter(Y_n).values()) # counts the elements' frequency