
"""
Created on Sun May 19 10:29:51 2019

@author
"""


from keras.applications.inception_v3 import InceptionV3
from keras import layers
from keras import models
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint
import json




#input_tensor = layers.Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'
#base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)
base_model = InceptionV3(weights='imagenet',
                         include_top=False,
                         input_shape = (224, 224, 3))
base_model.trainable = True

model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='softmax'))


model.summary()

from keras import optimizers

model.compile(loss = 'binary_crossentropy',
              optimizer = optimizers.RMSprop(lr=1e-4),
              metrics = ['acc'])

#Use ImageDataGenerator to preprocess images
train_dir = '/home/james/Dropbox/ML/Insight/derm_assist/data/ISIC_train'
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224,224),
        batch_size=1,
        class_mode='binary')



history = model.fit_generator(
        train_generator,
        steps_per_epoch = 100,
        epochs = 30,
        )



    
    
    
    