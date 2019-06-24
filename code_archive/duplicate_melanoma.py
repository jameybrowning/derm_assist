#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 15:25:35 2019

@author: james
"""

import os
import pandas as pd
import shutil
import random


dir_min = '/home/james/Dropbox/ML/Insight/derm_assist/data/train/pos'
dir_maj = '/home/james/Dropbox/ML/Insight/derm_assist/data/train/neg'

num_min = len(os.listdir(dir_min))
num_maj = len(os.listdir(dir_maj))
num_copies = int(num_maj/num_min) - 1
num_extra = num_maj % num_min

fnames=os.listdir(dir_min)

for i in range(1,num_copies+1):
    for j in range(0,num_min):
        print('copying image_'+str(j).zfill(5)+'_copy_'+str(i).zfill(2))
        fname1 = dir_min+'/'+fnames[j]
        fname2 = dir_min+'/copy_'+str(i).zfill(3)+'_'+fnames[j]
        shutil.copy(fname1, fname2)  
        
for i in range(0,num_extra): 
    print('copying extra image_'+str(i).zfill(6))
    radn_int =  random.randint(0,len(fnames))
    fname1 = dir_min+'/'+fnames[i]
    fname2 = dir_min+'/copy_extra_'+str(i).zfill(6)+'_'+fnames[i]
    shutil.copy(fname1, fname2)  
   

    
     
