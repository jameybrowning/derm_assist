#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:39:18 2019

@author: james
"""
import matplotlib.pyplot as plt

#adapted from:
#https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/
import numpy as np
fpr_keras = np.loadtxt('fpr_keras.txt')
tpr_keras = np.loadtxt('tpr_keras.txt')
thresholds = np.loadtxt('thresholds_keras.txt')

plt.figure(1)
#plt.plot([0, 1], [0, 1], 'k--')
#plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.plot(fpr_keras, tpr_keras)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
# Zoom in view of the upper left corner.
#plt.figure(2)
#plt.xlim(0, 0.2)
#plt.ylim(0.8, 1)
#plt.plot([0, 1], [0, 1], 'k--')
##plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
#plt.plot(fpr_keras, tpr_keras)
#plt.xlabel('False positive rate')
#plt.ylabel('True positive rate')
#plt.title('ROC curve (zoomed in at top left)')
#plt.legend(loc='best')
#plt.show()
