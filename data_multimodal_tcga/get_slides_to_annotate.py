# =============================================================================
# Code to select slides for manual annotation
# =============================================================================

import pandas as pd
import numpy as np

cases_per_classe=10

np.random.seed(seed=123)

df = pd.read_csv('patient-info-tcga.csv')

ind=pd.unique(df["subject_id"])

# for col in df.columns:
#     print(col)
    
class_0_m = df[(df['IDH1_mut']==0) & (df['loh1p/19q_cnv']==0) & (df['is_female']==0)]
class_1_m = df[(df['IDH1_mut']==1) & (df['loh1p/19q_cnv']==0) & (df['is_female']==0)]
class_2_m = df[(df['IDH1_mut']==1) & (df['loh1p/19q_cnv']==1) & (df['is_female']==0)]
class_0_f = df[(df['IDH1_mut']==0) & (df['loh1p/19q_cnv']==0) & (df['is_female']==1)]
class_1_f = df[(df['IDH1_mut']==1) & (df['loh1p/19q_cnv']==0) & (df['is_female']==1)]
class_2_f = df[(df['IDH1_mut']==1) & (df['loh1p/19q_cnv']==1) & (df['is_female']==1)]


class_all = [class_0_m,class_0_f, class_1_m,class_1_f, class_2_m,class_2_f]

cases=[]

for m in class_all:
    rd = np.random.randint(len(pd.unique(m["subject_id"])), size=int(cases_per_classe/2))
    sample = pd.unique(m["subject_id"])[rd]
    for idx in sample:
        cases.append(m[m["subject_id"]==idx].iloc[0,:])  
    
       

df_all = pd.DataFrame(cases)

df_all = df_all.filter(['index', 'subject_id','slide_id','is_female','age','IDH1_mut','loh1p/19q_cnv'])


df_all.to_excel("List_for_Dr_Claire_manual_ROI_annotation.xlsx",
             sheet_name='annotation_list') 