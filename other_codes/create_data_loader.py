# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:50:20 2022

@author: ge92tis
"""

import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib

example_filename = r'C:\Users\ge92tis\Desktop\TCGA-02-0003\flair.nii.gz'

img = nib.load(example_filename)
img_data = img.get_fdata()


print(np.mean(img_data))
print(np.max(img_data))
print(np.min(img_data))

import skimage.io as io
def show_volume_slice(data):
    io.imshow(data, cmap = 'gray')
    io.show()
    
show_volume_slice(img_data[:,145])

    
    # img = nib.load(image_modalilty[0])
    # img_flair = img.get_fdata()
    # img = nib.load(image_modalilty[1])
    # img_t1 = img.get_fdata()
    # img = nib.load(image_modalilty[2])
    # img_t1ce = img.get_fdata()
    # img = nib.load(image_modalilty[3])
    # img_t2 = img.get_fdata()