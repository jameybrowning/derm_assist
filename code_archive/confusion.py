#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 09:56:54 2019

@author: james
"""


#adapted from
#https://gist.github.com/RyanAkilos/3808c17f79e77c4117de35aa68447045
import numpy as np
from keras import backend as K
from keras import models
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

#Start
#train_data_path = 'F://data//Train'
val_dir = '/home/ubuntu/derm_assist/data/images_split_by_class/val'

#epochs = 30
batch_size = 32
#num_of_train_samples = 3000
#num_of_test_samples = 600

#Image Generator

datagen = ImageDataGenerator(rescale=1. / 255)


val_generator = datagen.flow_from_directory(val_dir,
                                                        target_size=(224, 224),
                                                        batch_size=batch_size,
                                                        shuffle = False,
                                                        class_mode='categorical')


'''
# Build model
model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(img_rows, img_cols, 3), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, (3, 3), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3, 3), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#Train
model.fit_generator(train_generator,
                    steps_per_epoch=num_of_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=num_of_test_samples // batch_size)
'''
model = models.load_model('derm_assist_checkpoint_drop_05.h5') 
#Confution Matrix and Classification Report
Y_pred = model.predict_generator(val_generator, val_generator.n // batch_size+1)
#y_pred = np.argmax(Y_pred, axis=1)
y_pred = np.rint(Y_pred)
print('Confusion Matrix')
print(confusion_matrix(val_generator.classes, y_pred))
print('Classification Report')
target_names = ['Mel', 'NonMel']
print(classification_report(val_generator.classes, y_pred, target_names=target_names))
