#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:49:40 2019

@author: james
"""
import os
import pandas as pd
import shutil
dir_main = '/home/james/Dropbox/ML/Insight/derm_assist/data/ISIC_2019_Training_Input/'

dir_neg = '/home/james/Dropbox/ML/Insight/derm_assist/data/ISIC_2019_neg/'
dir_pos = '/home/james/Dropbox/ML/Insight/derm_assist/data/ISIC_2019_pos/'

#os.makedirs('/home/james/Dropbox/ML/Insight/derm_assist/data/ISIC_2019_pos')

#os.makedirs('/home/james/Dropbox/ML/Insight/derm_assist/data/ISIC_2019_neg')

labels=pd.read_csv('/home/james/Dropbox/ML/Insight/derm_assist/data/ISIC_2019_Training_GroundTruth.csv')

for row in labels.itertuples():
    if row.MEL == 1.0:
        print(row)
        fname1 = dir_main+row.image+'.jpg'
        fname2 = dir_pos+row.image+'.jpg'
        shutil.move(fname1, fname2)  

    
     

    



