#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:01:44 2019

@author: james
"""
from keras import models
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


val_dir = '/home/ubuntu/derm_assist/data/images_split_by_class/val'
model_name = 'resnet'
batch_size = 32
model = models.load_model('networks/derm_assist_checkpoint_drop_05.h5') 

datagen = ImageDataGenerator(rescale=1. / 255)
val_generator = datagen.flow_from_directory(val_dir,
                                                        target_size=(224, 224),
                                                        batch_size=batch_size,
                                                        shuffle = False,
                                                        class_mode='categorical')
y_pred = model.predict_generator(val_generator, val_generator.n // batch_size+1)
y_gt = val_generator.classes
fname1 = 'y_pred_'+model_name
fname2 = 'y_gt_'+model_name
np.savetxt(fname1, y_pred)
np.savetxt(fname2, y_gt)