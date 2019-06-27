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


#load trained DermAssistAI model
model = models.load_model('networks/derm_assist_checkpoint_resnet_aug.h5') 
#model = models.load_model('derm_assist_checkpoint_MobileNetV2.h5') 
#model = models.load_model('derm_assist_checkpoint_drop_05.h5') 

# directories containing images to run through LIME
dir_derm = [None] * 2
dir_derm[0] = '/home/ubuntu/derm_assist/data/images_split_by_class/test/pos'
dir_derm[1] = '/home/ubuntu/derm_assist/data/images_split_by_class/test/neg'

#directories to save masks
dir_masks = [None]*2
dir_masks[0] = '/home/ubuntu/derm_assist/data/masks/pos'
dir_masks[1] = '/home/ubuntu/derm_assist/data/masks/neg'

# number of imaes to run from each class
num_images = 50
#number of combinations of each image to run through model to get LIME linear approximation
num_samples = 1000

#resize, add singleton dimension, and scale for input to CNN model
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

#loop over classes (positive and negative)
for i in range(2):
    #print(i)
    fnames = os.listdir(dir_derm[i])
    #select images randomly from directories
    rand_perm = np.random.permutation(len(fnames))
    
    #loop over images
    for j in range(num_images):
        fname = fnames[rand_perm[j]]
        print(fname[:-4])
        images = transform_img_fn([os.path.join(dir_derm[i],fname)])
    
        #get model prediction
        pred = model.predict(images)
        print(pred)    
    
        #run LIME
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(images[0], model.predict, top_labels=1, hide_color=0, num_samples=num_samples)
        image_int, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False)
        
        #produce and save LIME saliency masks
        plt.imshow(mark_boundaries(image_int, mask))
        np.save(dir_masks[i]+'/'+fname[:-4]+'_mask', mask)
        np.save(dir_masks[i]+'/'+fname[:-4]+'_image_int', image_int)
 
    





