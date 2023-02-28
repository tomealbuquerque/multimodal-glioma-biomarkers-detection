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
parser.add_argument('--tile_size', type=int, choices=[224,512,1024],default=512)
parser.add_argument('--threshold_otsu_tiles', type=float ,default=0.4)
parser.add_argument('--folds', type=int, choices=range(5),default=5)
parser.add_argument('--seed', type=int,default=1234)
parser.add_argument('--name_file', type=str, default="multimodal_glioma_data_full2.pickle")
args = parser.parse_args()


import numpy as np
import pandas as pd
import glob
import pickle
import os
from tiles_generator import get_tiles
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold


#read dataframe with the patients data
df_clean = pd.read_csv('patient-info-tcga.csv')


# #get unique values from dataframe (remove histology duplicates)
# a = df['subject_id'].unique()

# #create a new dataframe with all the multimodal data avaiable
# df_clean = df.drop_duplicates(subset=["subject_id"])

# #reset index from dataframe
# df_clean = df_clean.reset_index()


# =============================================================================
# Prepare MRI + histology + Clinical data
# =============================================================================

image_modalilty=[]
X_full_data=[]
Y_full_data=[]
X_patient_data=[]
Y_patient_data=[]
i=0
idx=0


while len(df_clean)>idx:
    print('slide:',idx,'/',len(df_clean)-1)
    if idx<2:
        image_modalilty=[]
        for image_modal in glob.glob("Radiology\\" +str(df_clean.loc[idx,'subject_id']) +"\\*"):
            image_modalilty.append(image_modal)

        X_patient_data.append({'patient':df_clean.loc[idx,'subject_id'],
                            'flair':image_modalilty[0],
                            't1':image_modalilty[1],
                            't1ce':image_modalilty[2],
                            't2':image_modalilty[3],
                            'slide':df_clean.loc[idx,'slide_id'],
                            'tiles_coords':get_tiles(slide_path=os.path.join("E:\\Pathology",df_clean.loc[idx,'slide_id']),
                                                      tile_size=args.tile_size, threshold_for_otsu=args.threshold_otsu_tiles),
                            'gender':df_clean.loc[idx,'is_female'],
                            'age':df_clean.loc[idx,'age']})#
        

        Y_patient_data.append({'idh1':int(df_clean.loc[idx,'IDH1_mut']),'ioh1p19q':int(df_clean.loc[i,'loh1p/19q_cnv'])})
        
        idx=idx+1
        

    if idx>=2:
        while (idx>1)  & (df_clean.loc[idx,'subject_id'] == df_clean.loc[idx-1,'subject_id']):

            image_modalilty=[]
            for image_modal in glob.glob("Radiology\\" +str(df_clean.loc[idx,'subject_id']) +"\\*"):
                image_modalilty.append(image_modal)
                
    
            X_patient_data.append({'patient':df_clean.loc[idx,'subject_id'],
                                'flair':image_modalilty[0],
                                't1':image_modalilty[1],
                                't1ce':image_modalilty[2],
                                't2':image_modalilty[3],
                                'slide':df_clean.loc[idx,'slide_id'],
                                'tiles_coords':get_tiles(slide_path=os.path.join("E:\\Pathology",df_clean.loc[idx,'slide_id']),
                                                          tile_size=args.tile_size, threshold_for_otsu=args.threshold_otsu_tiles),
                                'gender':df_clean.loc[idx,'is_female'],
                                'age':df_clean.loc[idx,'age']})#
            
       
            Y_patient_data.append({'idh1':int(df_clean.loc[idx,'IDH1_mut']),'ioh1p19q':int(df_clean.loc[idx,'loh1p/19q_cnv'])})
            idx=idx+1
               
        else:
            X_patient_data=[]
            Y_patient_data=[]
            image_modalilty=[]
            for image_modal in glob.glob("Radiology\\" +str(df_clean.loc[idx,'subject_id']) +"\\*"):
                image_modalilty.append(image_modal)
                
    
            X_patient_data.append({'patient':df_clean.loc[idx,'subject_id'],
                                'flair':image_modalilty[0],
                                't1':image_modalilty[1],
                                't1ce':image_modalilty[2],
                                't2':image_modalilty[3],
                                'slide':df_clean.loc[idx,'slide_id'],
                                'tiles_coords':get_tiles(slide_path=os.path.join("E:\\Pathology",df_clean.loc[idx,'slide_id']),
                                                          tile_size=args.tile_size, threshold_for_otsu=args.threshold_otsu_tiles),
                                'gender':df_clean.loc[idx,'is_female'],
                                'age':df_clean.loc[idx,'age']})#
            
            
       
            Y_patient_data.append({'idh1':int(df_clean.loc[idx,'IDH1_mut']),'ioh1p19q':int(df_clean.loc[idx,'loh1p/19q_cnv'])})
            
            idx=idx+1
   
    X_full_data.append(X_patient_data)
    Y_full_data.append(Y_patient_data)
    

X = np.array(X_full_data, dtype=object)
Y = np.array(Y_full_data, dtype=object)


# =============================================================================
# Create Kfolds for cross-validation
# =============================================================================

# from sklearn.utils.multiclass import type_of_target
# type_of_target(Y)

state = np.random.RandomState(args.seed)
kfold = KFold(args.folds, shuffle=True, random_state=state)
folds = [{'train': (X[tr], Y[tr]), 'test': (X[ts], Y[ts])} 
    for tr, ts in kfold.split(X, Y)]
pickle.dump(folds, open(args.name_file, 'wb'))


# lib,target = folds[0]['train']

# lib_f=[]
# target_f=[]
# for i in range(len(lib)):
#     for z in range(len(lib[i])):
#         lib_f.append(lib[i][z])
#         target_f.append(target[i][z])
    