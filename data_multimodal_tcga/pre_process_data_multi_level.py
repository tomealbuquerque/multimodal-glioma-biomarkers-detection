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
parser.add_argument('--tile_size', type=int, choices=[224, 512], default=512)
parser.add_argument('--tile_size_level', type=int, choices=[1024, 2048], default=2048)
parser.add_argument('--threshold_otsu_tiles', type=float, default=0.4)
parser.add_argument('--folds', type=int, choices=range(5), default=5)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--name_file', type=str, default="multimodal_glioma_data_multi_level.pickle")
args = parser.parse_args()

import numpy as np
import pandas as pd
import glob
import pickle
import os
from tiles_generator import get_tiles
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold

# read dataframe with the patients data
df = pd.read_csv('patient-info-tcga.csv')
ds_base_path = 'Radiology'

# get unique values from dataframe (remove histology duplicates)
a = df['subject_id'].unique()

# create a new dataframe with all the multimodal data avaiable
df_clean = df.drop_duplicates(subset=["subject_id"])

# reset index from dataframe
df_clean = df_clean.reset_index()

# =============================================================================
# Prepare MRI + histology + Clinical data
# =============================================================================

image_modality = []
X_full_data = []
Y_full_data = []

for idx, pat in enumerate(df_clean['subject_id']):
    image_modality = []

    ds_folders_path = os.path.join(ds_base_path, str(pat), "*")
    for image_modal in glob.glob(ds_folders_path):
        image_modality.append(image_modal)

    """   
    #create array of dictionares with input data per patient
    {
    't1ce': str (path_to_image),
    't2': str (path_to_image),
    'flair':str (path_to_image),
    't2_block':str (path_to_image),
    't1': str (path_to_image),
    't1_block': str (path_to_image),
    'flair_block': str (path_to_image),
    't1ce_block': str (path_to_image),
    'slide': str (path_to_image),
    'tiles_coords': list of tuples (int,int)
    'gender': int,
    'age': int
    }
    
    """
    X_full_data.append({'t1ce': image_modality[0],
                        't2': image_modality[1],
                        'flair': image_modality[2],
                        't2_block': image_modality[3],
                        't1': image_modality[4],
                        't1_block': image_modality[5],
                        'flair_block': image_modality[6],
                        't1ce_block': image_modality[7],
                        'slide': os.path.join("Pathology", df_clean.loc[idx, 'slide_id']),
                        'tiles_coords': get_tiles(slide_path=os.path.join("Pathology", df_clean.loc[idx, 'slide_id']),
                                                  tile_size=args.tile_size,
                                                  threshold_for_otsu=args.threshold_otsu_tiles),
                        'gender': df_clean.loc[idx, 'is_female'],
                        'age': df_clean.loc[idx, 'age']})  #

    """   
    #create array of dictionares with target data per patient
    {
    'idh1' int, 
    'ioh1p19q': int
    }
    
    """
    Y_full_data.append({'idh1': int(df_clean.loc[idx, 'IDH1_mut']),
                        'ioh1p19q': int(df_clean.loc[idx, 'loh1p/19q_cnv'])})

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
