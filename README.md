# Multimodal Context-Aware Detection of Glioma Biomarkers using MRI and WSI

by Tomé Albuquerque, Mei Ling Fang, Benedikt Wiestler, Claire Delbridge, Maria João M. Vasconcelos, Jaime S. Cardoso and
Peter Schüffler

<div align="center">Schematic representation of the different models used in this work:<br />
<img src="https://github.com/tomealbuquerque/multimodal-glioma-biomarkers-detection/blob/main/Figures/scheme.PNG" width="600"></div>

## <div align="center">Documentation</div>
<details Open>
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

## <div align="center">1) Pre-processing</div>

First, let's create a "data.pickle" with an array of dictionaries containing all the data information from MRI and WSI for the training and test process for 5-folds. The dictionary will have the following structure:

```
X[fold] = {
            'flair':str (path_to_image),
            't1': str (path_to_image),
            't1ce': str (path_to_image),
            't2': str (path_to_image),
            'flair_block':str (path_to_image),
            't1_block': str (path_to_block),
            't1ce_block': str (path_to_block),
            't2_block': str (path_to_block),
            'slide': str (path_to_slide),
            'tiles_coords': list of tuples (int,int)
            'tiles_coords_level': list of tuples (int,int)
            'gender': int, 
            'age': int
          }
  
Y[fold] = {
            'idh1' int, 
            'ioh1p19q': int
          }
```
 
To create run: 
```
python data_multimodal_tcga/pre_process_data_multi_level.py
```  
P.S.: The "data.pickle" is provided in this repo for TCGA dataset.
</details>

## <div align="center">2) Train embedder (MRI and WSI)</div>

###MRI embedder
missing information...

###WSI embedder

To train the emmbeder for 552x512 tiles run the following command:
```
python WSI_embedder\MIL_512_tiles\mil_train_bin.py --fold 0 --model 'checkpoint_best_512_bin_fold_0.pth'
```
for 2048x2048 tiles change just the path: WSI_embedder\MIL_2048_tiles\mil_train_bin.py


You can skip this step and use the provided pre-trained models weights for WSI and for MRI:
| **fold** | **WSI** | **Original - MRI T1ce** | **Original - MRI - T1Ce + FLAIR** | **Segmented - MRI - T1ce** | **Segmented - MRI - T1ce + FLAIR** |
|:--------:|:-------:|:-----------------------:|:---------------------------------:|:--------------------------:|:----------------------------------:|
|   **0**  |   [x]   |           [x]           |                [x]                |             [x]            |                 [x]                |
|   **1**  |   [x]   |           [x]           |                [x]                |             [x]            |                 [x]                |
|   **2**  |   [x]   |           [x]           |                [x]                |             [x]            |                 [x]                |
|   **3**  |   [x]   |           [x]           |                [x]                |             [x]            |                 [x]                |
|   **4**  |   [x]   |           [x]           |                [x]                |             [x]            |                 [x]                |
