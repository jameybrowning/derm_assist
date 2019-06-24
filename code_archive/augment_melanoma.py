#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 13:55:53 2019

@author: james
"""

import os
import random
from scipy import ndarray

# image processing library
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io
from skimage import img_as_ubyte



def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree, resize = True)


#def random_zoom(image_array: ndarray):
#    #pick random zoom range <= 1. 1 returns original image
#    random_zoom = random.uniform(.75, 1)
#    ymin = int(image_array.shape[1]*((1-random_zoom)/2))
#    ymax = int(image_array.shape[1]*(1-(1-random_zoom)/2))
#    xmin = int(image_array.shape[2]*((1-random_zoom)/2))
#    xmax = int(image_array.shape[2]*(1-(1-random_zoom)/2))
#    return image_array[ymin:ymax,xmin:xmax]

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

# dictionary of the transformations we defined earlier
available_transformations = {
    'rotate': random_rotation,
#    'zoom': random_zoom,
    'noise': random_noise,
    'horizontal_flip': horizontal_flip
}

folder_path = '/home/james/Dropbox/ML/Insight/derm_assist/data/train/pos'
folder_path2 = '/home/james/Dropbox/ML/Insight/derm_assist/data/ISIC_2019_pos_aug'
num_files_desired = 30

# find all files paths from the folder
images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

num_generated_files = 0
while num_generated_files <= num_files_desired:
    print(num_generated_files)
    # random image from the folder
    image_path = random.choice(images)
    # read image as an two dimensional array of pixels
    image_to_transform = sk.io.imread(image_path)
    # random num of transformation to apply
    num_transformations_to_apply = random.randint(1, len(available_transformations))

    num_transformations = 0
    transformed_image = None
    while num_transformations <= num_transformations_to_apply:
        # random transformation to apply for a single image
        key = random.choice(list(available_transformations))
        transformed_image = available_transformations[key](image_to_transform)
        num_transformations += 1

    new_file_path = '%s/augmented_image_%s.jpg' % (folder_path2, num_generated_files)
    
    # write image to the disk
    transformed_image_uint8 = img_as_ubyte(transformed_image)
    io.imsave(new_file_path, transformed_image_uint8)
    num_generated_files += 1