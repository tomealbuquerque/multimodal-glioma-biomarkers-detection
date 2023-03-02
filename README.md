# Multimodal Context-Aware Detection of Glioma Biomarkers using MRI and WSI

by Tomé Albuquerque, Mei Ling Fang, Benedikt Wiestler, Claire Delbridge, Maria João M. Vasconcelos, Jaime S. Cardoso and
Peter Schüffler

The most malignant tumors of the central nervous system are adult-type diffuse gliomas. Historically, glioma classification has been based on morphological features. However, since 2016, WHO recognizes molecular evaluation to be critical for the subtype classification.  Among molecular markers, the mutation status of the IDH1 and the codeletion of 1p/19q are crucial for the precise diagnosis of these malignancies. In pathological labs, manual screening is time-consuming and susceptible to error. To overcome these limitations, we propose a novel multimodal biomarker classification method that, integrates image features derived from brain Magnetic resonance imaging (MRI) and histopathology exams (WSI). The proposed model is composed of two branches, the first branch takes as input a multi-scale Hematoxylin and Eosin (H&E) whole slide image, and the second the pre-segmented region-of-interest from the MRI. Both branches are based on Convolutional Neural Networks (CNN). After passing the exams by the two embedding branches the output feature vectors are concatenated and using a multi-layer perceptron the glioma biomarkers are classified based on a multi-class problem. Several fusion strategies were studied in this work including cascade model with mid fusion; mid fusion model; late fusion model, and mid context fusion. The models were tested using a publicly available data set from The Cancer Genome Atlas (TCGA). The overall cross-validated classification obtained area under the curve (AUC) of 87.48%, 86.32%, and 81.54% for multimodality, MRI, and H&E stain slide images respectively, outperforming their unimodal counterparts as well as the state-of-the-art methods in classifying glioma biomarkers.

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


## <div align="center">2) Train embedder (MRI and WSI)</div>

* MRI embedder

missing information...


* WSI embedder


To train the emmbeder for 552x512 tiles run the following command:
```
python WSI_embedder\MIL_512_tiles\mil_train_bin.py --fold 0
```
for 2048x2048 tiles change just the path: WSI_embedder\MIL_2048_tiles\mil_train_bin.py

After training the model it is necessary to generate a list of all tiles per slide with the output probabilities (e.g."predictions_grid_{typee}_fold_{args.fold}_bin.csv"), for that run:
```
python WSI_embedder\MIL_512_tiles\MIL_get_GRIDS.py --fold 0 --model 'checkpoint_best_512_bin_fold_0.pth'
```

#You can skip the training of the embedders and use the provided pre-trained models weights for WSI and for MRI:

| **fold** | **WSI** | **Original - MRI T1ce** | **Original - MRI - T1Ce + FLAIR** | **Segmented - MRI - T1ce** | **Segmented - MRI - T1ce + FLAIR** |
|:--------:|:-------:|:-----------------------:|:---------------------------------:|:--------------------------:|:----------------------------------:|
|   **0**  |   [x]   |           [x]           |                [x]                |             [x]            |                 [x]                |
|   **1**  |   [x]   |           [x]           |                [x]                |             [x]            |                 [x]                |
|   **2**  |   [x]   |           [x]           |                [x]                |             [x]            |                 [x]                |
|   **3**  |   [x]   |           [x]           |                [x]                |             [x]            |                 [x]                |
|   **4**  |   [x]   |           [x]           |                [x]                |             [x]            |                 [x]                |


## <div align="center">3) Train/test multimodal agregator (MRI + WSI)</div>

</details>
