"""
# =============================================================================
# Preprocessing code for Multimodal Glioma biomarkers prognosis using TCGA dataset

# X-> MRI('FLAIR', 'T2', 'T1', 'T1Gd'); WSI(file_name); clinical_data(gender,age)
# Y-> biomarkers ('IDH1_mut', 'loh1p/19q_cnv)

# K-folds generation for cross-validation purposes

# Tomé Albuquerque
# =============================================================================
"""


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--folds', type=int, choices=range(5),default=5)
parser.add_argument('--seed', type=int,default=1234)
parser.add_argument('--name_file', type=str, default="multimodal_glioma_data.pickle")
args = parser.parse_args()


import numpy as np
import pandas as pd
import glob
import pickle
import os
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold


#read dataframe with the patients data
df = pd.read_csv('patient-info-tcga.csv')


#get unique values from dataframe (remove histology duplicates)
a = df['subject_id'].unique()

#create a new dataframe with all the multimodal data avaiable
df_clean = df.drop_duplicates(subset=["subject_id"])

#reset index from dataframe
df_clean = df_clean.reset_index()


# =============================================================================
# Prepare MRI + histology + Clinical data
# =============================================================================

image_modalilty=[]
X_full_data=[]
Y_full_data=[]

for idx,pat in enumerate(df_clean['subject_id']):
    print(pat)
    image_modalilty=[]
    for image_modal in glob.glob("Radiology\\" +str(pat) +"\\*"):
        image_modalilty.append(image_modal)
        
    """   
    #create array of dictionares with input data per patient
    {
    'flair':str (path_to_image),
    't1': str (path_to_image),
    't1ce': str (path_to_image),
    't2': str (path_to_image),
    'slide': str (path_to_image),
    'gender': int, 
    'age': int
    }
    
    """
    X_full_data.append({'flair':image_modalilty[0],'t1':image_modalilty[1],'t1ce':image_modalilty[2],'t2':image_modalilty[3],
                        'slide':df_clean.loc[idx,'slide_id'], 'gender':df_clean.loc[idx,'is_female'], 'age':df_clean.loc[idx,'age']})#
    
    
    """   
    #create array of dictionares with target data per patient
    {
    'idh1' int, 
    'ioh1p15q': int
    }
    
    """    
    Y_full_data.append({'idh1':int(df_clean.loc[idx,'IDH1_mut']),'ioh1p15q':int(df_clean.loc[idx,'loh1p/19q_cnv'])})


X = np.array(X_full_data)
Y = np.array(Y_full_data)


# =============================================================================
# Create Kfolds for cross-validation
# =============================================================================

from sklearn.utils.multiclass import type_of_target
type_of_target(Y)

state = np.random.RandomState(args.seed)
kfold = KFold(args.folds, shuffle=True, random_state=state)
folds = [{'train': (X[tr], Y[tr]), 'test': (X[ts], Y[ts])} 
    for tr, ts in kfold.split(X, Y)]
pickle.dump(folds, open(args.name_file, 'wb'))