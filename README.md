## Multimodal Context-Aware Detection of Glioma Biomarkers using MRI and WSI

by Tomé Albuquerque, Mei Ling Fang, Benedikt Wiestler, Claire Delbridge, Maria João M. Vasconcelos, Jaime S. Cardoso and
Peter Schüffler

<div align="center">Schematic representation of the different models used in this work:<br />
<img src="https://github.com/tomealbuquerque/multimodal-glioma-biomarkers-detection/blob/main/Figures/scheme.PNG" width="600"></div>

## <div align="center">Documentation</div>
<details Close>
<summary>Requirements</summary>

* Image==1.5.33
* monai==1.0.0
* opencv_python_headless==4.5.5.62
* openslide_python==1.2.0
* nibabel==5.0.1
* Pillow==9.4.0
* scikit_image==0.19.2
* scikit_learn==1.2.1
* seaborn==0.11.2
* skimage==0.0
* torch==1.10.0
* torchvision==0.11.1
  
```
pip install -r requirements.txt
```
</details>

<details Open>
<summary>Usage</summary>
<b>1) Pre-processing<b>
  
Run: 
```
python data_multimodal_tcga/pre_process_data_multi_level.py
```  
  
</details>
