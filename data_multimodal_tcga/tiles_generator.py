"""
# =============================================================================
# Code for geting Tiles from a .svs file - with Otsu threshold to remove  the background

# Tomé Albuquerque
# =============================================================================
"""
import os
os.add_dll_directory('C:\\Users\\ge92tis\\openslide-win64-20220811\\bin')
import openslide
from openslide.deepzoom import DeepZoomGenerator
import cv2
import numpy as np
from skimage import filters
from PIL import Image
from openslide import *



def get_tiles(slide_path, tile_size=512, threshold_for_otsu=0.4):
    
    '''
    input -> slide_path, tile_size, threshold_for_otsu
    
    return -> list of coordinates of the Tiles
    
    '''

    threshold = threshold_for_otsu
    
    slide = openslide.OpenSlide(slide_path)
    
    data_gen = DeepZoomGenerator(slide, tile_size=tile_size, overlap=0, limit_bounds=False)
    
    max_level= data_gen.level_count-1
    x=data_gen.level_tiles[max_level]
    
    cords_deep_zoom, coords_512 = [], []
    count_512 = 0
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
    grid_loc_512, coords_512 = [], []
    count_512 = 0
    patch_size  = tile_size
    
    # tile_size = patch_size - 2*overlap  
    tile_size = tile_size
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
        
                    grid_loc_512.append((x,y))
                    coords_512.append(tile_coords)
                    count_512 += 1
    
            x += 1              
        y += 1
        x = 0
    
   
    return coords_512

