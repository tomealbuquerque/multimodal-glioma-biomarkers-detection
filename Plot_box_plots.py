# =============================================================================
# code to generate tables from results\ .txt files
# =============================================================================

import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_dataframe(path, mix, loss,name,metric):
    for i in ([mix]):
        for l in ([loss]):
            for s in ([10]):
                acc=[]
                bacc=[]
                mse=[]
                f1=[]
                auc=[]
                precision=[]
                for f in range(5):
    
                    text_file = open(os.path.join(path,f"results_{s}_mix_{i}_loss_{l}_fold_{f}.txt"))
                    
                    lines = text_file.readlines()
                    acc.append(float(lines[3].split(":")[1]))
                    mse.append(float(lines[4].split(":")[1]))
                    f1.append(float(lines[5].split(":")[1]))
                    precision.append(float(lines[7].split(":")[1]))
                    auc.append(float(lines[8].split(":")[1]))
                    bacc.append(float(lines[9].split(":")[1]))
                    
    if metric=="Accuracy (%)":
        met=acc        
    elif metric=="AUC (%)":
        met=auc
    elif metric=="F1 (%)":
        met=f1
    else:
        pass
    
        
    dframe_cross=[]           
    for i in range(5):
        dframe_cross.append([name, i, met[i]])
    
    df = pd.DataFrame(dframe_cross,columns=['Model', 'fold_idx', metric])

    return df


def get_datafram_MRI(data,name,metric):
    
    dframe_cross=[]           
    for i in range(5):
        dframe_cross.append([name, i, data[i]])
    
    dfm = pd.DataFrame(dframe_cross,columns=['Model', 'fold_idx', metric])
    
    return dfm

# =============================================================================
# Plot the boxplot
# =============================================================================

def get_full_datframe(metric="AUC (%)"):
    if metric=="Accuracy (%)":
        MRI_t1ce =  [0.7632, 0.8421, 0.7568, 0.8108, 0.8378]
        MRI_t1ce_flair= [0.7632, 0.7368, 0.7162, 0.6892, 0.7568]
        MRI_t1ce_original = [i/100 for i in [57.895, 65.789, 72.973, 67.568, 64.865]]
        MRI_t1ce_flair_original = [i/100 for i in [65.789, 65.789, 75.676, 70.270, 64.865]]
    elif metric=="AUC (%)":
        MRI_t1ce =  [0.8565426628, 0.8586716524, 0.8439452704, 0.8714786756,0.8857555328]
        MRI_t1ce_flair = [0.8899334357, 0.8579415954, 0.8337867198, 0.7882505612, 0.8338240314]
        MRI_t1ce_original = [i/100 for i in [58.640, 66.425, 72.008, 67.794, 66.436]]
        MRI_t1ce_flair_original = [i/100 for i in [72.779, 71.202, 66.195, 76.734, 69.757]]
    elif metric=="F1 (%)":
        MRI_t1ce =  [0.7629, 0.8406, 0.7541, 0.8125, 0.8465]
        MRI_t1ce_flair= [0.7599, 0.7014, 0.6976, 0.5907, 0.690]
        MRI_t1ce_original = [i/100 for i in [50.220, 59.610, 72.650, 61.650, 57.170]]
        MRI_t1ce_flair_original = [i/100 for i in [62.280, 61.120, 74.170, 66.300, 54.980]]
        
    
    
    
    df_mri_t1ce_flair_original = get_datafram_MRI(MRI_t1ce_flair_original, "MRI - T1ce\FLAIR Original ", metric)
    df_mri_t1ce_original = get_datafram_MRI(MRI_t1ce_original,"MRI - T1ce Original ",metric)
    df_mri_t1ce_flair = get_datafram_MRI(MRI_t1ce_flair, "MRI - T1ce\FLAIR Segment. ", metric)
    df_mri_t1ce = get_datafram_MRI(MRI_t1ce,"MRI - T1ce Segment. ",metric)
    
    df_HE_1 = get_dataframe(path=r"28-Partial_dataset_MIL_bin_multiclass_kfold\results",
                         mix="global", loss="CE",name="H&E - global",metric=metric)
    df_HE_2 = get_dataframe(path=r"28-Partial_dataset_MIL_bin_multiclass_kfold\results",
                            mix="mix", loss="CE",name="H&E - mix",metric=metric)
    df_HE_3 = get_dataframe(path=r"28-Partial_dataset_MIL_bin_multiclass_kfold\results",
                         mix="expected", loss="CE",name="H&E - expected",metric=metric)
 
    
    # df_multi_1 = get_dataframe(path=r"7-Partial_dataset_MIL_bin_MRI_multi_t1ce_flair_cascade_concat_1024\results", mix="mix",
    #                      loss="ordinal",name="H&E + T1ce\FLAIR - Cascade - OE",metric=metric)
    df_multi_2 = get_dataframe(path=r"34-Partial_dataset_MIL_bin_MRI_multi_t1ce_flair_cascade_concat_reg_CE\results", mix="expected",
                         loss="CE",name="H&E + T1ce/FLAIR - Cascade ",metric=metric)
    
    
    # df_multi_3 = get_dataframe(path=r"9-Partial_dataset_MIL_bin_MRI_multi_t1ce_flair_fusion\results", mix="mix",
    #                      loss="ordinal",name="H&E + T1ce/FLAIR - Mid Fusion - OE",metric=metric)
    df_multi_4 = get_dataframe(path=r"31-Partial_dataset_MIL_bin_MRI_multi_t1ce_flair_fusion_CE\results", mix="expected",
                         loss="CE",name="H&E + T1ce/FLAIR - Mid Fusion ",metric=metric)
    
    df_multi_5 = get_dataframe(path=r"35-Partial_dataset_MIL_bin_MRI_multi_t1ce_flair_fusion_CE_level\results", mix="expected",
                         loss="CE",name="H&E + T1ce/Flair - Mid  Context Fusion ",metric=metric)
    
    df_multi_55 = get_dataframe(path=r"36-Partial_dataset_MIL_bin_MRI_multi_t1ce_fusion_CE_level\results", mix="expected",
                         loss="CE",name="H&E + T1ce - Mid  Context Fusion ",metric=metric)
    
    # df_multi_5 = get_dataframe(path=r"21-Partial_dataset_MIL_bin_MRI_multi_t1ce_flair_fusion_late_ordinal\results", mix="mix",
    #                      loss="ordinal",name="H&E + T1ce/FLAIR - Late Fusion - OE",metric=metric)
    df_multi_6 = get_dataframe(path=r"29-Partial_dataset_MIL_bin_MRI_multi_t1ce_flair_late_fusion_CE\results", mix="expected",
                         loss="CE",name="H&E + T1ce/FLAIR - Late Fusion ",metric=metric)
    
    # df_multi_7 = get_dataframe(path=r"8-Partial_dataset_MIL_bin_MRI_multi_t1ce_cascade_concat_1024\results", mix="mix",
    #                      loss="ordinal",name="H&E + T1ce - Cascade - OE",metric=metric)
    df_multi_8 = get_dataframe(path=r"33-Partial_dataset_MIL_bin_MRI_multi_t1ce_cascade_concat_reg_CE\results", mix="expected",
                         loss="CE",name="H&E + T1ce - Cascade ",metric=metric)
    
    
    # df_multi_9 = get_dataframe(path=r"24-Partial_dataset_MIL_bin_MRI_multi_t1ce_fusion_ordinal\results", mix="mix",
    #                      loss="ordinal",name="H&E + T1ce - Mid Fusion - OE",metric=metric)
    df_multi_10 = get_dataframe(path=r"32-Partial_dataset_MIL_bin_MRI_multi_t1ce_fusion_CE\results", mix="expected",
                         loss="CE",name="H&E + T1ce - Mid Fusion ",metric=metric)
    

    # df_multi_11 = get_dataframe(path=r"18-Partial_dataset_MIL_bin_MRI_multi_t1ce_fusion_late_ordinal\results", mix="mix",
    #                      loss="ordinal",name="H&E + T1ce - Late Fusion - OE",metric=metric)
    df_multi_12 = get_dataframe(path=r"30-Partial_dataset_MIL_bin_MRI_multi_t1ce_late_fusion_CE\results", mix="expected",
                         loss="CE",name="H&E + T1ce - Late Fusion ",metric=metric)
    
    df_all = pd.concat((df_HE_1,
                        df_HE_2,
                        df_HE_3,
                        # df_HE_4,
                        # df_HE_5,
                        # df_HE_6,
                        df_mri_t1ce_flair_original,
                        df_mri_t1ce_original,
                        df_mri_t1ce_flair,
                        df_mri_t1ce,
                        # df_multi_1,
                        df_multi_8,
                        # df_multi_9,
                        df_multi_10,
                        # df_multi_11,
                        df_multi_12,
                        df_multi_55,
                        df_multi_2,
                        # df_multi_3,
                        df_multi_4,
                        df_multi_6,
                        df_multi_5
                        
                        # df_multi_7
                        ),axis=0)


    return df_all

df_all_auc = get_full_datframe()
df_all_acc = get_full_datframe("Accuracy (%)")
df_all_f1 = get_full_datframe("F1 (%)")

# sns.set(style="darkgrid")
sns.set_theme(style="ticks")
# https://medium.com/swlh/how-to-create-a-seaborn-palette-that-highlights-maximum-value-f614aecd706b
colors= [plt.cm.tab10(i) for i in range(30)]
palette = np.vstack(([colors[6] for i in range(3)],[colors[7] for i in range(4)],[colors[0] for i in range(8)]))


a4_dims = (10, 4)
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=a4_dims)  # choose appropriate size to fit your needs
box_bigrams_AUC=sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
box_bigrams_AUC=sns.boxplot(x="AUC (%)", y='Model', data=df_all_auc, palette=palette, showfliers = True, ax=ax1)
box_bigrams_ACC=sns.boxplot(x="Accuracy (%)", y='Model', data=df_all_acc, palette=palette, showfliers = False, ax=ax2)
box_bigrams_F1=sns.boxplot(x="F1 (%)", y='Model', data=df_all_f1, palette=palette, showfliers = False, ax=ax3)

ax2.get_yaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)
ax1.xaxis.grid(True)
ax2.xaxis.grid(True)
ax3.xaxis.grid(True)

box_bigrams_ACC.set_xlabel(box_bigrams_ACC.get_xlabel(),fontsize=14, fontweight='bold')
box_bigrams_F1.set_xlabel(box_bigrams_F1.get_xlabel(),fontsize=14, fontweight='bold')
box_bigrams_AUC.set_xlabel(box_bigrams_AUC.get_xlabel(),fontsize=14, fontweight='bold')
box_bigrams_AUC.set_ylabel(box_bigrams_AUC.get_ylabel(),fontsize=14, fontweight='bold')
fig.savefig('box_plot_all.png', bbox_inches='tight', dpi=200)


# a4_dims = (3, 7)
# fig, ax = plt.subplots(figsize=a4_dims)

# # ax.annotate('global', xy=(0.05, -0.1), xytext=(0.05, -0.2),
# #             fontsize=14, ha='center', va='bottom', xycoords='axes fraction', 
# #             bbox=dict(boxstyle='square', fc='0.8'),
# #             arrowprops=dict(arrowstyle='-[, widthB=2.0, lengthB=.5', lw=2.0))

# ax.xaxis.grid(True)
# sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
# box_bigrams = sns.boxplot(ax=ax, x="AUC (%)", y='Model', data=df_all_auc, palette=palette, showfliers = False)
# # box_bigrams.set_xticklabels(box_bigrams.get_xticklabels(),rotation=90)

# box_bigrams.set_xlabel(box_bigrams.get_xlabel(),fontsize=14, fontweight='bold')
# box_bigrams.set_ylabel(box_bigrams.get_ylabel(),fontsize=14, fontweight='bold')
# ax.xaxis.grid(True)
# fig_bigrams = box_bigrams.get_figure()
# fig_bigrams.savefig('boxplot_bigrams.png')