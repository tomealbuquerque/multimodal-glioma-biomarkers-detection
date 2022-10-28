# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:44:45 2022

@author: albu
"""

import urllib
url, filename = ("https://github.com/mateuszbuda/brain-segmentation-pytorch/raw/master/assets/TCGA_CS_4944.png", "TCGA_CS_4944.png")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)


import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import pickle
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import os
import matplotlib.pyplot as plt

model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)




X,Y = pickle.load(open(os.path.join('data_multimodal_tcga',f'multimodal_glioma_data.pickle'), 'rb'))[0]['train']

img = nib.load(os.path.join('data_multimodal_tcga',X[0]['flair']))
img_flair = img.get_fdata()


im = Image.fromarray(img_flair[:,:,40])
im = im.convert('RGB')

# input_image = Image.open(filename)
input_image = im
m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=m, std=s),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model = model.to('cuda')

with torch.no_grad():
    output = model(input_batch)

print(torch.round(output[0]))




image=torch.round(output[0]).cpu().numpy()

image = image[0,:, :]

imgplot = plt.imshow(image)

# imgplot = plt.imshow(img_flair[:,:,40])
