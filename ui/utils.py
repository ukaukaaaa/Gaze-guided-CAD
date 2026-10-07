import numpy as np
import pylink

def truncate_hu(image_array, mode='soft_tissue'):
    newimg = image_array.copy()
    if mode == 'soft_tissue':
        newimg[image_array > 225] = 225
        newimg[image_array <-125] = -125
    elif mode == 'lung':
        newimg[image_array > 200] = 200
        newimg[image_array <-1400] = -1400
    elif mode == 'bone':
        newimg[image_array > 1500] = 1500
        newimg[image_array <-500] = -500
    
    return newimg
    
def normalazation(image_array):
    max = image_array.max()
    min = image_array.min()
    newimg = (image_array - min)/(max - min)
    newimg = (newimg*255).astype(np.uint8)
    return newimg 
