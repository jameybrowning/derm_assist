#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 22:54:52 2019

@author: james
"""

import os
import keras
from keras.applications import inception_v3 as inc_net
from keras.applications.mobilenetv2 import MobileNetV2
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
print('Notebook run using keras:', keras.__version__)

model = inc_net.InceptionV3()
#model = inc_net(
#        include_top=True,
#        weights='imagenet',
#        #input_tensor=input_tensor,
#        input_shape=(224, 224, 3))
#pooling='avg')  

def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = inc_net.preprocess_input(x)
        out.append(x)
    return np.vstack(out)

#images = transform_img_fn([os.path.join('data','cat_mouse.jpg')])
images = transform_img_fn([os.path.join('LIME_data','hen.jpg')])
# I'm dividing by 2 and adding 0.5 because of how this Inception represents images
plt.imshow(images[0] / 2 + 0.5)
preds = model.predict(images)
for x in decode_predictions(preds)[0]:
    print(x)
    
from lime import lime_image

explainer = lime_image.LimeImageExplainer()

explanation = explainer.explain_instance(images[0], model.predict, top_labels=5, hide_color=0, num_samples=500)

from skimage.segmentation import mark_boundaries

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=1, hide_rest=True)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))



#temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=1, hide_rest=False)
#plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))



