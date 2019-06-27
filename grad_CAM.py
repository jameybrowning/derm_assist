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


model = models.load_model('../checkpoints/derm_assist_checkpoint_resnet_aug.h5') 

model.summary()

def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x/255
        #x = model.preprocess_input(x)
        out.append(x)
    return np.vstack(out)

img_path = '/home/james/Dropbox/ML/Insight/derm_assist/data/ISIC_2019_all/ISIC_0026930.jpg'
img = image.load_img(img_path,target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img/255                        
preds = model.predict(img)
                        
import keras.backend as K


# This is the "african elephant" entry in the prediction vector
output_mel = model.output[0]

# The is the output feature map of the `block5_conv3` layer,
# the last convolutional layer in VGG16
last_conv_layer = model.get_layer('activation_49')

# This is the gradient of the "african elephant" class with regard to
# the output feature map of `block5_conv3`
grads = K.gradients(output_mel, last_conv_layer.output)[0]

# This is a vector of shape (512,), where each entry
# is the mean intensity of the gradient over a specific feature map channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))

# This function allows us to access the values of the quantities we just defined:
# `pooled_grads` and the output feature map of `block5_conv3`,
# given a sample image
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

# These are the values of these two quantities, as Numpy arrays,
# given our sample image of two elephants
pooled_grads_value, conv_layer_output_value = iterate([img])

# We multiply each channel in the feature map array
# by "how important this channel is" with regard to the elephant class
for i in range(np.size(conv_layer_output_value,2)):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# The channel-wise mean of the resulting feature map
# is our heatmap of class activation
heatmap = np.mean(conv_layer_output_value, axis=-1)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()

import cv2

# We use cv2 to load the original image
img = cv2.imread(img_path)

# We resize the heatmap to have the same size as the original image
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

# We convert the heatmap to RGB
heatmap = np.uint8(255 * heatmap)

# We apply the heatmap to the original image
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 0.4 here is a heatmap intensity factor
superimposed_img = heatmap * 0.4 + img


plt.imshow(superimposed_img/np.max(superimposed_img))
