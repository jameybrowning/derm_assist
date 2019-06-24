#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:50:14 2019

@author: james
"""
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt

plt.close("all")
 
model_name1 = 'resnet_aug_test'
model_name2 = 'resnet_test'
model_name3 = 'mobilenet_test'


#https://github.com/scikit-learn/scikit-learn/blob/master/examples/model_selection/plot_confusion_matrix.py

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize = 20)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    fig.tight_layout()
    return ax

y_pred1 = np.loadtxt('../performance/y_pred_'+model_name1+'.txt')
y_gt1 = np.loadtxt('../performance/y_gt_'+model_name1+'.txt')

y_pred2 = np.loadtxt('../performance/y_pred_'+model_name2+'.txt')
y_gt2 = np.loadtxt('../performance/y_gt_'+model_name2+'.txt')

y_pred3 = np.loadtxt('../performance/y_pred_'+model_name3+'.txt')
y_gt3 = np.loadtxt('../performance/y_gt_'+model_name3+'.txt')

y_pred_bin1 = np.rint(y_pred1)
y_pred_bin2 = np.rint(y_pred2)
y_pred_bin3 = np.rint(y_pred3)

plot_confusion_matrix(y_gt1, y_pred_bin1, classes=["Mel","Other"], normalize = True,
                      title='ResNet50')

plot_confusion_matrix(y_gt2, y_pred_bin2, classes=["Mel","Other"], normalize = True,
                      title='ResNet50')

plot_confusion_matrix(y_gt3, y_pred_bin3, classes=["Mel","Other"], normalize = True,
                      title='MobileNetV2')

fpr1, tpr1, thresholds1 = roc_curve(y_gt1, y_pred1)
fpr2, tpr2, thresholds2 = roc_curve(y_gt2, y_pred2)
fpr3, tpr3, thresholds3 = roc_curve(y_gt3, y_pred3)

tnr1 = 1-fpr1
tnr2 = 1-fpr2
tnr3 = 1-fpr3

auc1 = auc(fpr1, tpr1)
auc2 = auc(fpr2, tpr2)
auc3 = auc(fpr3, tpr3)

plt.figure()
#plt.plot([0, 1], [0, 1], 'k--')
plt.plot(tnr1, tpr1, label='ResNet50 (AUC = {:.3f})'.format(auc1))
plt.plot(tnr2, tpr2, label='ResNet50 (AUC = {:.3f})'.format(auc2))
plt.plot(tnr3, tpr3, label='MobileNetV2 (AUC = {:.3f})'.format(auc3))
plt.xlabel('True negative rate (specificity)', fontsize = 20)
plt.ylabel('True positive rate (sensitivity)', fontsize = 20)
plt.title('ROC curve', fontsize = 20)
plt.legend(loc='best')
plt.legend(fontsize = 18)
plt.ylim((0,1))
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.show()

plt.figure()
#plt.plot([0, 1], [0, 1], 'k--')
plt.plot(tnr1, tpr1, label='ResNet50 (AUC = {:.3f})'.format(auc1))
plt.plot(tnr2, tpr2, label='ResNet50 (AUC = {:.3f})'.format(auc2))
plt.plot(tnr3, tpr3, label='MobileNetV2 (AUC = {:.3f})'.format(auc3))
plt.xlabel('True negative rate (specificity)', fontsize = 20)
plt.ylabel('True positive rate (sensitivity)', fontsize = 20)
plt.title('ROC curve', fontsize = 20)
plt.legend(loc='best')
plt.legend(fontsize = 18)
plt.ylim((0.9,1))
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.show()








