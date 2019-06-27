#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
trains DermAssistAI model from pretrained ResNet30 with imagenet weights

'''


import PIL
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenetv2 import MobileNetV2
from keras import layers
from keras import models
from keras import callbacks
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

image_size = 224
batch_size = 32
epochs = 50
epsilon = 0.25
gamma = 2

base_model = ResNet50(weights='imagenet',
                             include_top=False,
                             input_shape = (image_size,image_size, 3))
base_model.trainable = True
x = base_model.output
x = layers.Flatten()(x)
x = layers.Dropout(0.25)(x)
x = layers.Dense(256, activation = 'relu')(x)
y = layers.Dense(1, activation='sigmoid')(x)
model = models.Model(inputs = base_model.input, outputs = y)
 

#model = models.load_model('networks/derm_assist_checkpoint_resnet_aug.h5')


#model.compile(loss = 'binary_crossentropy',
#              optimizer = optimizers.RMSprop(lr=1e-4),
#              metrics = ['acc'])

#Use ImageDataGenerator to preprocess images


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    #shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
	)

val_datagen = ImageDataGenerator(rescale=1./255)

train_dir = '/home/ubuntu/derm_assist/data/images_split_by_class/train'
val_dir = '/home/ubuntu/derm_assist/data/images_split_by_class/val'
#train_dir = '/home/james/Dropbox/ML/Insight/derm_assist/data/train'
#val_dir = '/home/james/Dropbox/ML/Insight/derm_assist/data/val'

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size,image_size),
        batch_size=batch_size,
        class_mode='binary',
	shuffle = True)

val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(image_size,image_size),
        batch_size=batch_size,
        class_mode='binary')

callbacks_list = [callbacks.ModelCheckpoint(
        filepath = 'derm_assist_focal_checkpoint_resnet_bestacc.h5',
        monitor = 'val_acc',
        save_best_only = True), callbacks.ModelCheckpoint(
        filepath = 'derm_assist_focal_checkpoint_resnet_bestloss.h5',
        monitor = 'val_loss',
        save_best_only = True),
callbacks.CSVLogger('log_resnet_focal.csv', append=True, separator=';')]       

import keras.backend as K

def custom_loss(epsilon, gamma):
    
    def loss(y_true, y_pred):
        pt = y_pred * y_true + (1-y_pred) * (1-y_true)
        pt = K.clip(pt, epsilon, 1-epsilon)
        CE = -K.log(pt)
        FL = K.pow(1-pt, gamma) * CE
        focal_loss = K.sum(FL, axis=1)
            
        return focal_loss
    
    return loss

#model.compile(optimizer= optimizers.RMSprop(lr=1e-4), 
#              loss=custom_loss(epsilon, gamma), metrics = ['acc'])

model.compile(loss = 'binary_crossentropy',
              optimizer = optimizers.RMSprop(lr=1e-4),
              metrics = ['acc'])
history = model.fit_generator(
        train_generator,
        steps_per_epoch = train_generator.n//batch_size,
        epochs = epochs,
        validation_data=val_generator,
        validation_steps=val_generator.n//batch_size,
        callbacks = callbacks_list)








