#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:16:18 2019

@author: james
"""
from matplotlib import pyplot as plt
import os
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.transform import rescale, resize, downscale_local_mean
dir_derm = '/home/james/Dropbox/ML/Insight/derm_assist/code/LIME_data/derm_neg'

image_num = 0
fnames = os.listdir(dir_derm)
image1 = plt.imread(os.path.join(dir_derm,fnames[image_num]))

image2 = resize(image1, (224, 224),
                       anti_aliasing=True)
segments_quick = quickshift(image2, kernel_size=4, max_dist=200, ratio=0.2)
segments_fz = felzenszwalb(image2, scale=100, sigma=0.5, min_size=50)

plt.imshow(mark_boundaries(image2, segments_quick))
plt.imshow(mark_boundaries(image2, segments_fz))


