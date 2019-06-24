#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:51:30 2019

@author: james
"""
import os
import numpy as np
import shutil
dir_main = '/home/james/Dropbox/ML/Insight/derm_assist/data'
dir_pos_all = '/home/james/Dropbox/ML/Insight/derm_assist/data/ISIC_2019_pos'
dir_neg_all = '/home/james/Dropbox/ML/Insight/derm_assist/data/ISIC_2019_neg'

#create train directories

dirs = [None]*9
dirs[0] = os.path.join(dir_main, 'train')
dirs[1] = os.path.join(dir_main, 'val')
dirs[2] = os.path.join(dir_main, 'test')

dirs[3] = os.path.join(dirs[0], 'pos')
dirs[4] = os.path.join(dirs[0], 'neg')

dirs[5] = os.path.join(dirs[1], 'pos')
dirs[6] = os.path.join(dirs[1], 'neg')

dirs[7] = os.path.join(dirs[2], 'pos')
dirs[8] = os.path.join(dirs[2], 'neg')


for i in dirs:
    os.mkdir(i)

    
#move 80, 10, 10 percent of positive to train, val, test respectively
split1_fract = .8
split2_fract = .9
#get list of files in positive directory
fnames_pos_all = os.listdir(dir_pos_all)
split1_idx= int(split1_fract*len(fnames_pos_all))
split2_idx= int(split2_fract*len(fnames_pos_all))
#random permuation of indices
rand_perm = np.random.permutation(len(fnames_pos_all))
#copy images
print('moving positive training images')
for i in range(0,split1_idx):    #train
    fname1 = dir_pos_all+'/'+fnames_pos_all[rand_perm[i]]
    fname2 = dirs[3]+'/'+fnames_pos_all[rand_perm[i]]
    shutil.copy(fname1, fname2)
print('moving positive val images')  
for i in range(split1_idx,split2_idx):      #val
    fname1 = dir_pos_all+'/'+fnames_pos_all[rand_perm[i]]
    fname2 = dirs[5]+'/'+fnames_pos_all[rand_perm[i]]
    shutil.copy(fname1, fname2)
print('moving positive test images')   
for i in range(split2_idx,len(fnames_pos_all)):     #test
    fname1 = dir_pos_all+'/'+fnames_pos_all[rand_perm[i]]
    fname2 = dirs[7]+'/'+fnames_pos_all[rand_perm[i]]
    shutil.copy(fname1, fname2)
    
    
    
    
split3_idx = (split2_idx-split1_idx)  #validation
split4_idx = split3_idx + len(fnames_pos_all)-split2_idx #test
fnames_neg_all = os.listdir(dir_neg_all)
rand_perm = np.random.permutation(len(fnames_neg_all))
print('moving negative val images')
for i in range(0,split3_idx):    #val
    fname1 = dir_neg_all+'/'+fnames_neg_all[rand_perm[i]]
    fname2 = dirs[6]+'/'+fnames_neg_all[rand_perm[i]]
    shutil.copy(fname1, fname2)
print('moving negative  test images')   
for i in range(split3_idx,split4_idx):      #test
    fname1 = dir_neg_all+'/'+fnames_neg_all[rand_perm[i]]
    fname2 = dirs[8]+'/'+fnames_neg_all[rand_perm[i]]
    shutil.copy(fname1, fname2)
print('moving negative  train images')    
for i in range(split4_idx,len(fnames_neg_all)):     #train
    fname1 = dir_neg_all+'/'+fnames_neg_all[rand_perm[i]]
    fname2 = dirs[4]+'/'+fnames_neg_all[rand_perm[i]]
    shutil.copy(fname1, fname2)





    
   

