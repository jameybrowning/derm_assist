#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:01:44 2019

@author: james

saves predictions and ground truths for trained model
"""
from keras import models
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

#directory containing images to run inference on
val_dir = '/home/ubuntu/derm_assist/data/images_split_by_class/test'
#name for saved files
model_name = 'resnet_aug_test'

batch_size = 32
model = models.load_model('models/derm_assist_checkpoint_resnet_aug.h5') 

datagen = ImageDataGenerator(rescale=1. / 255)

val_generator = datagen.flow_from_directory(val_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    shuffle = False,
    class_mode='categorical')

y_pred = model.predict_generator(val_generator, val_generator.n // batch_size+1)
y_gt = val_generator.classes

fname1 = 'y_pred_'+model_name+'.txt'
fname2 = 'y_gt_'+model_name+'.txt'

np.savetxt(fname1, y_pred)
np.savetxt(fname2, y_gt)
