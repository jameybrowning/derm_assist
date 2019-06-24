#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 09:34:53 2019

@author: james
"""

# load network
from keras.models import load_model
import pandas as pd
import math
from keras_preprocessing.image import ImageDataGenerator
model = load_model('/home/james/Dropbox/ML/Insight/derm_assist/checkpoints/da_resnet_50_checkpoint.h5')




train_df = pd.read_csv('/home/james/Dropbox/ML/Insight/derm_assist/data/train_df.csv')
num_images= train_df.shape[0]
split = 0.9
split_index = math.floor(split*num_images)
val_datagen=ImageDataGenerator(rescale=1./255.)
val_dataframe = train_df[split_index:]
val_dataframe = val_dataframe.reset_index(drop = True)
val_generator=val_datagen.flow_from_dataframe(
dataframe=val_dataframe,
#directory='/home/james/Dropbox/ML/Insight/derm_assist/data/ISIC_2019_Training_Input',
directory= '/home/james/Dropbox/ML/Insight/derm_assist/data/ISIC_2019_all',
x_col="image",
#has_ext=True,
batch_size=1,
seed=42,
shuffle=False,
class_mode=None,
target_size=(224,224))


STEP_SIZE_VAL=val_generator.n//val_generator.batch_size

val_generator.reset()
pred=model.predict_generator(val_generator,
steps=STEP_SIZE_VAL,
verbose=1)


pred_bool = (pred >0.5)

predictions = pred_bool.astype(int)



results=pd.DataFrame(predictions, columns=columns)
results["Filenames"]=test_generator.filenames
ordered_cols=["Filenames"]+columns
results=results[ordered_cols]#To get the same column order
results.to_csv("results.csv",index=False)
 
