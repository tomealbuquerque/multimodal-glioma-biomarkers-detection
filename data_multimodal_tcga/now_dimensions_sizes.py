# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 14:19:36 2022

@author: albu
"""


import os
#os.add_dll_directory('C:\\Users\\ge92tis\\openslide-win64-20220811\\bin')
import openslide
from openslide.deepzoom import DeepZoomGenerator
import cv2
import numpy as np
from skimage import filters
from PIL import Image
from openslide import *


slide = openslide.OpenSlide('E:\\Pathology\\TCGA-02-0003-01Z-00-DX1.6171b175-0972-4e84-9997-2f1ce75f4407.svs')

data_gen = DeepZoomGenerator(slide, tile_size=512, overlap=0, limit_bounds=False)

max_level= data_gen.level_count-1
x=data_gen.level_tiles[max_level]

# cords_deep_zoom, coords_512 = [], []
count_512 = 0
x, y = 0, 0

x_tiles, y_tiles = data_gen.level_tiles[max_level]
slide_dims = slide.dimensions