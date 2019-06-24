#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 12:47:05 2019

@author: james
"""
import os
import numpy as np
from keras import models
import fnmatch
from keras.preprocessing import image
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries
from PIL import Image

plt.close("all")

model = models.load_model('../checkpoints/derm_assist_checkpoint_resnet_aug.h5') 
dir_all = '/home/james/Dropbox/ML/Insight/derm_assist/data/ISIC_2019_all'

dir_masks = [None]*2
dir_masks[0] = '/home/james/Dropbox/ML/Insight/derm_assist/data/masks/neg'
dir_masks[1] = '/home/james/Dropbox/ML/Insight/derm_assist/data/masks/pos'

lime_dir = [None]*2
lime_dir[0] = '/home/james/Dropbox/ML/Insight/derm_assist/data/lime_images/neg'
lime_dir[1] = '/home/james/Dropbox/ML/Insight/derm_assist/data/lime_images/pos'




for i in range(2):
    fnames1 = os.listdir(dir_masks[i])   
    fnames_masks = fnmatch.filter(fnames1, '*mask.npy')
    
    for j in range(len(fnames_masks)):
        plt.close("all")
        num_images = len(fnames_masks)
        print(j)
        mask = np.load(dir_masks[i]+'/'+fnames_masks[j])
        lime_image = np.load(dir_masks[i]+'/'+fnames_masks[j][:-9]+'_image_int.npy')
        full_image = Image.open(dir_all+'/'+fnames_masks[j][:-9]+'.jpg')
        sqrWidth = np.ceil(np.sqrt(full_image.size[0]*full_image.size[1])).astype(int)
        full_image = full_image.resize((sqrWidth, sqrWidth))
        
        img = image.load_img(dir_all+'/'+fnames_masks[j][:-9]+'.jpg', target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        im_scaled = x/255
        
        pred = model.predict(im_scaled)
      
        plt.figure(figsize=(30,15))
        plt.subplot(1, 2, 1)
        plt.imshow(full_image)
        plt.title('Original Image', fontsize = 24)
        
        true_class = i
        plt.subplot(1,2, 2)
        plt.imshow(mark_boundaries(lime_image, mask))
        plt.title('LIME Result (Risk = {:.3f}, True Class = {:.0f})'.format(np.float64(pred),true_class), fontsize = 24)
        
        
        plt.savefig(lime_dir[i]+'/'+fnames_masks[j][:-9]+'_lime_image.jpg')
        
