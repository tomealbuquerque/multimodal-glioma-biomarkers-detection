# -*- coding: utf-8 -*-
"""
# =============================================================================
# Code to visualize the patients clinical data - get some stats 
# =============================================================================

@author: Tome Albuquerque
"""
#important imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#read dataframe with the patients data
df = pd.read_csv('patient-info-tcga.csv')

#get the name of columns
data_top = list(df.columns)
print(data_top)


#print some hist plots      about important features

fig = plt.figure()

#Three integers (nrows, ncols, index)
ax1 = fig.add_subplot(221)
age_val=len(df.age.unique())

plot_age= df["age"].plot(kind='hist' , ax=ax1, bins=round(age_val/2),rwidth=0.7,range=[min(df.age.unique()),max(df.age.unique())])
plt.title('Patients Age')

ax2 = fig.add_subplot(222) 
survival_months_val=len(df.survival_months.unique())

df["survival_months"].plot(kind='hist' , bins=round(survival_months_val/2), ax=ax2)
plt.title('survival_months')


ax3 = fig.add_subplot(223) 
df["is_female"].plot(kind='hist',ax=ax3)
plt.title('Male vs Female')

ax4 = fig.add_subplot(224) 
df["IDH1_mut"].plot(kind='hist',ax=ax4)
plt.title('IDH1 mutation')

fig.tight_layout(pad=1.0)
plt.savefig("stats.png", dpi=300, bbox_inches = 'tight')
plt.show()


#get specific correlations
print('correlation between age and survival months:', df['age'].corr(df['survival_months']))
print('correlation between IDH1_mut and survival months:',df['IDH1_mut'].corr(df['survival_months']))
print('correlation between gender and survival months:',df['is_female'].corr(df['survival_months']))
print('correlation between loh1p/19q_cnv and survival months:',df['loh1p/19q_cnv'].corr(df['survival_months']))


#get all correlations in a matrix
df_feat=df.drop(columns=['index', 'subject_id', 'slide_id', 'FLAIR', 'T2', 'T1', 'T1Gd', 'oncotree_code', 'train','censorship'])
corr = df_feat.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(df_feat.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90, fontsize=6)
plt.yticks(fontsize=6)
ax.set_yticks(ticks)
ax.set_xticklabels(df_feat.columns)
ax.set_yticklabels(df_feat.columns)
plt.savefig("correlation.png", dpi=300, bbox_inches = 'tight')
plt.show()

