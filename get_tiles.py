# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 10:10:36 2022

@author: ge92tis
"""

import os
os.add_dll_directory('C:\\Users\\ge92tis\\openslide-win64-20220811\\bin')
import openslide
import pandas as pd
from openslide.deepzoom import DeepZoomGenerator
import cv2
import numpy as np
from skimage import filters
from PIL import Image
from openslide import *


slide_number=1
threshold = 0.4

list_slides=[]
list_slide_name=[]

path = "test_slides"
for x in os.listdir(path):
    list_slides.append(os.path.join(path,x))
    list_slide_name.append(x)
    
d_sli = {'slide':list_slide_name}
dataframe_slides = pd.DataFrame(data=d_sli)
dataframe_slides.to_csv('list_slides.csv', index=False)

slide = openslide.OpenSlide(list_slides[slide_number])

#size of the image
print(slide.level_dimensions[0])


data_gen = DeepZoomGenerator(slide, tile_size=512, overlap=0, limit_bounds=False)

max_level= data_gen.level_count-1
x=data_gen.level_tiles[max_level]


cords_deep_zoom, coords_512 = [], []
count_224 = 0
x, y = 0, 0

x_tiles, y_tiles = data_gen.level_tiles[max_level]
slide_dims = slide.dimensions

down_factor = int(slide.level_downsamples[-1])
isup = 0

# Generate thumbnail file
slide_thumb = slide.get_thumbnail((int(np.floor(slide_dims[0]/down_factor)),int(np.floor(slide_dims[1]/down_factor))))

# Apply Otsu's threshold to remove background
hsv = cv2.cvtColor(np.array(slide_thumb), cv2.COLOR_BGR2HSV)
_, s, _ = cv2.split(hsv)
s = cv2.GaussianBlur(s ,(15,15),0)
val = filters.threshold_otsu(s)
_, thumb_otsu = cv2.threshold(s , val, 255, cv2.THRESH_BINARY)        
thumbnail_otsu = Image.fromarray(np.uint8(thumb_otsu))

# Generate tiles for .svs & thumbnail files
grid_loc_224, coords_512 = [], []
count_224 = 0
patch_size  = 512

# tile_size = patch_size - 2*overlap  
tile_size = 512
tiles_svs = DeepZoomGenerator(slide, tile_size, 0)
tiles_otsu = DeepZoomGenerator(ImageSlide(thumbnail_otsu), tile_size/down_factor, 0) 

max_level_svs = tiles_svs.level_count - 1    
max_level_otsu = tiles_otsu.level_count - 1 

# Tiles sorting & labeling
x, y = 0, 0
x_tiles, y_tiles = tiles_otsu.level_tiles[max_level_otsu]


while y < y_tiles:
    while x < x_tiles:  
        with np.errstate(divide='ignore'):                                         
            new_tile_otsu = np.array(tiles_otsu.get_tile(max_level_otsu, (x, y)), dtype=np.uint8)
        
            if (np.sum(new_tile_otsu, axis=2) == 0).sum() <= np.round((1-threshold)*(tile_size/down_factor)**2):
        
                new_tile_svs = np.array(tiles_svs.get_tile(max_level_svs, (x, y)), dtype=np.uint8)
                tile_coords = tiles_svs.get_tile_coordinates(max_level_svs, (x, y))[0]
    
                grid_loc_224.append((x,y))
                coords_512.append(tile_coords)
                count_224 += 1

        x += 1              
    y += 1
    x = 0

d = {'cords':coords_512}
df = pd.DataFrame(data=d)
df.to_csv('cord_tilles_dim_512_slide_'+str(slide_number)+'.csv', index=True)

reg = slide.read_region(coords_512[1], 0, (224, 224)).convert('RGB').save('test.jpg')