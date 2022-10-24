# =============================================================================
# code to generate tables from results\ .txt files
# =============================================================================

import numpy as np

for m in (["t1","t1ce","t2","flair"]):
#for s in ([3, 5]):
    acc=[]
    mse=[]
    f1=[]
    precision=[]
    for i in range(5):
        text_file = open("results\\architecture-resnext50_32x4d-method-UniMRI-MRI_type-"+str(m)+"-fold-"+str(i)+"-epochs-100-batchsize-32-lr-0.0001.txt", "r")
        
      #architecture-resnext50_32x4d-method-UniMRI-MRI_type-flair-fold-2-epochs-100-batchsize-32-lr-0.0001 
        lines = text_file.readlines()
        acc.append(float(lines[3].split(":")[1]))
        mse.append(float(lines[4].split(":")[1]))
        f1.append(float(lines[5].split(":")[1]))
        precision.append(float(lines[7].split(":")[1]))
       
    accm=np.mean(acc)
    msem=np.mean(mse)
    f1m=np.mean(f1)
    precisionm=np.mean(precision)
    accd=np.std(acc)
    msed=np.std(mse)
    f1d=np.std(f1)
    precisiond=np.std(precision)
    
    
    print("\\textbf{"+str(m.upper())+"} & $"+'{:.3f}'.format(round(accm*100, 3))+" \pm "+ '{:.3f}'.format(round(accd*100, 3))+"$ & $"+
          '{:.5f}'.format(round(msem,5))+" \pm "+ '{:.5f}'.format(round(msed, 5))+"$ & $"+
          '{:.3f}'.format(round(f1m*100,3))+" \pm "+ '{:.3f}'.format(round(f1d*100, 3))+"$ & $"+
          '{:.3f}'.format(round(precisionm*100, 3))+" \pm "+ '{:.3f}'.format(round(precisiond*100, 3))+"$")
    
 