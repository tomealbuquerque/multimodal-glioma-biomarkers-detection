# =============================================================================
# code to generate tables from results\ .txt files
# =============================================================================

import numpy as np
import os


folder_path=r"35-Partial_dataset_MIL_bin_MRI_multi_t1ce_flair_fusion_CE_level\results"



# 7-Partial_dataset_MIL_bin_MRI_multi_t1ce_flair_cascade_concat_1024\results
# 13-Partial_dataset_MIL_bin_MRI_multi_t1ce_flair_fusion_late_CE\results

#'7-Partial_dataset_MIL_bin_MRI_multi_t1ce_flair_cascade_concat_1024'
#'9-Partial_dataset_MIL_bin_MRI_multi_t1ce_flair_fusion'
#'21-Partial_dataset_MIL_bin_MRI_multi_t1ce_flair_fusion_late_ordinal'
#'8-Partial_dataset_MIL_bin_MRI_multi_t1ce_cascade_concat_1024'
#'24-Partial_dataset_MIL_bin_MRI_multi_t1ce_fusion_ordinal'
#'18-Partial_dataset_MIL_bin_MRI_multi_t1ce_fusion_late_ordinal' 


for i in (["expected"]):
    print(i)
    for l in (["CE"]):
        print(l)
        for s in ([10]):
            acc=[]
            bacc=[]
            mse=[]
            f1=[]
            auc=[]
            precision=[]
            for f in range(5):

                text_file = open(os.path.join(folder_path,f"results_{s}_mix_{i}_loss_{l}_fold_{f}.txt"))
                
                 
                #architecture-resnext50_32x4d-method-UniMRI-MRI_type-flair-fold-2-epochs-100-batchsize-32-lr-0.0001 
                lines = text_file.readlines()
                acc.append(float(lines[3].split(":")[1]))
                mse.append(float(lines[4].split(":")[1]))
                f1.append(float(lines[5].split(":")[1]))
                precision.append(float(lines[7].split(":")[1]))
                auc.append(float(lines[8].split(":")[1]))
                bacc.append(float(lines[9].split(":")[1]))
                
                
            aucm=np.mean(auc)
            accm=np.mean(acc)
            baccm=np.mean(bacc)
            msem=np.mean(mse)
            f1m=np.mean(f1)
            precisionm=np.mean(precision)
            accd=np.std(acc)
            baccd=np.std(bacc)
            msed=np.std(mse)
            f1d=np.std(f1)
            precisiond=np.std(precision)
            aucd=np.std(auc)
                  


            print("\\textbf{"+str(s)+"} & $"+'{:.3f}'.format(round(accm*100, 3))+" \pm "+ '{:.3f}'.format(round(accd*100, 3))+"$ & $"+
            # '{:.3f}'.format(round(baccm*100,3))+" \pm "+ '{:.3f}'.format(round(baccd*100, 3))+"$ & $"+
            '{:.3f}'.format(round(msem,3))+" \pm "+ '{:.3f}'.format(round(msed, 3))+"$ & $"+
            '{:.3f}'.format(round(f1m*100,3))+" \pm "+ '{:.3f}'.format(round(f1d*100, 3))+"$ & $"+
            '{:.3f}'.format(round(precisionm*100, 3))+" \pm "+ '{:.3f}'.format(round(precisiond*100, 3))+"$ & $"+
            '{:.3f}'.format(round(aucm*100, 3))+" \pm "+ '{:.3f}'.format(round(aucd*100, 3))+"$\\\\")
            
            # if l=='ordinal':
            #     # print("&&\\textbf{"+str(s)+"} & $"+'{:.3f}'.format(round(accm*100, 3))+"$ & $"+
            #     # '{:.5f}'.format(round(msem,5))+"$ & $"+
            #     # '{:.3f}'.format(round(f1m*100,3))+"$ & $"+
            #     # '{:.3f}'.format(round(precisionm*100, 3))+"$ & $-$\\\\")
            #     print("\\textbf{"+str(s)+"} & $"+'{:.3f}'.format(round(accm*100, 3))+" \pm "+ '{:.3f}'.format(round(accd*100, 3))+"$ & $"+
            #     '{:.3f}'.format(round(baccm*100,3))+" \pm "+ '{:.3f}'.format(round(baccd*100, 3))+"$ & $"+
            #     '{:.5f}'.format(round(msem,3))+" \pm "+ '{:.3f}'.format(round(msed, 5))+"$ & $"+
            #     '{:.3f}'.format(round(f1m*100,3))+" \pm "+ '{:.3f}'.format(round(f1d*100, 3))+"$ & $"+
            #     '{:.3f}'.format(round(precisionm*100, 3))+" \pm "+ '{:.3f}'.format(round(precisiond*100, 3))+"$ & $-$\\\\")
            # else:
            # print("&&\\textbf{"+str(s)+"} & $"+'{:.3f}'.format(round(accm*100, 3))+"$ & $"+
            # '{:.5f}'.format(round(msem,5))+"$ & $"+
            # '{:.3f}'.format(round(f1m*100,3))+"$ & $"+
            # '{:.3f}'.format(round(precisionm*100, 3))+"$ & $"+
            # '{:.3f}'.format(round(aucm*100, 3))+"$\\\\")
