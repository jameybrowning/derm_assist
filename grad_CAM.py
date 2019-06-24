#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 15:23:18 2019

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

model.summary()

dir_derm = [None] * 2
dir_derm[0] = '/home/james/Dropbox/ML/Insight/derm_assist/code/LIME_data/derm_neg'
dir_derm[1] = '/home/james/Dropbox/ML/Insight/derm_assist/code/LIME_data/derm_pos'

image_num = 0
pos_neg = 1


