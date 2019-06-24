#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import PIL
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenetv2 import MobileNetV2
from keras import layers
from keras import models
from keras import callbacks

image_size = 224
batch_size = 32
epochs = 50

base_model = ResNet50(weights=None,
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
from keras import optimizers

model.compile(loss = 'binary_crossentropy',
              optimizer = optimizers.RMSprop(lr=1e-4),
              metrics = ['acc'])

#Use ImageDataGenerator to preprocess images
from keras.preprocessing.image import ImageDataGenerator

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
        filepath = 'derm_assist_checkpoint_resnet_448.h5',
        monitor = 'val_loss',
        save_best_only = True), callbacks.CSVLogger('log_resnet_448.csv', append=True, separator=';')]       


history = model.fit_generator(
        train_generator,
        steps_per_epoch = train_generator.n//batch_size,
        epochs = epochs,
        validation_data=val_generator,
        validation_steps=val_generator.n//batch_size,
        callbacks = callbacks_list)








