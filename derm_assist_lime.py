#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:13:20 2019

@author: james
"""

import numpy as np
import os
from keras import models
from keras.preprocessing import image
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries, quickshift, felzenszwalb, slic
from lime import lime_image
import time


model = models.load_model('derm_assist_checkpoint_resnet_aug.h5') 
#model = models.load_model('derm_assist_checkpoint_MobileNetV2.h5') 
#model = models.load_model('derm_assist_checkpoint_drop_05.h5') 

dir_derm = [None] * 2
dir_derm[0] = '/home/ubuntu/derm_assist/data/images_split_by_class/test/pos'
dir_derm[1] = '/home/ubuntu/derm_assist/data/images_split_by_class/test/neg'

#dir_derm[0] = '/home/james/Dropbox/ML/Insight/derm_assist/code/LIME_data/derm_pos'
#dir_derm[1] = '/home/james/Dropbox/ML/Insight/derm_assist/code/LIME_data/derm_neg'

dir_masks = [None]*2
dir_masks[0] = '/home/ubuntu/derm_assist/data/masks/pos'
dir_masks[1] = '/home/ubuntu/derm_assist/data/masks/neg'

#dir_masks[0] = '/home/james/Dropbox/ML/Insight/derm_assist/code/LIME_data/masks/pos'
#dir_masks[1] = '/home/james/Dropbox/ML/Insight/derm_assist/code/LIME_data/masks/neg'

# number of imaes to run from each class
num_images = 50
num_samples = 1000


def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x/255
        #x = model.preprocess_input(x)
        out.append(x)
    return np.vstack(out)
 
fnames = os.listdir(dir_derm[i])
    
images = transform_img_fn([os.path.join(dir_derm[pos_neg],fnames[image_num])])

pred = model.predict(images)

    





