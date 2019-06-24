#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os, shutil
import PIL

train_dir = '/home/ubuntu/derm_assist/data/images_split_by_class/train'

validation_dir = '/home/ubuntu/derm_assist/data/images_split_by_class/val'
from keras.applications.resnet50 import ResNet50
from keras import layers
from keras import models


#base_model = ResNet50(weights='imagenet',
#                             include_top=False,
#                             input_shape = (224, 224, 3))
#base_model.trainable = True
#x = base_model.output
##x = layers.Dropout(0.2)(x)
#x = layers.Flatten()(x)
#    #x = layers.Dropout(0.2)(x)
#x = layers.Dense(256, activation = 'relu')(x)
#y = layers.Dense(1, activation='sigmoid')(x)
#model = models.Model(inputs = base_model.input, outputs = y)
 
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation= 'relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

from keras import optimizers

model.compile(loss = 'binary_crossentropy',
              optimizer = optimizers.RMSprop(lr=1e-4),
              metrics = ['acc'])

#Use ImageDataGenerator to preprocess images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224,224),
        batch_size=1,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(224,224),
        batch_size=1,
        class_mode='binary')
        
#fit the model using a batch generator
history = model.fit_generator(
        train_generator,
        steps_per_epoch = 20,
        epochs = 50,
        validation_data=validation_generator,
        validation_steps=20)

#save model




