#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:59:48 2019

@author: simakovde
"""
import numpy as np
import matplotlib.pyplot as plt
import os

def block(ax, data_train, data_val, title, ylabel, xlabel, font_size=20):
    ax.plot(data_train, label='train', c='navy', alpha=0.6)
    ax.plot(data_val, label='val', c='skyblue')
    ax.set_title(title, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.legend(frameon=False, fontsize=font_size)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def vizualize(model_path, fold, show=True, save=False, save_dir=None, save_format='.pdf'):
    train_res = np.load(os.path.join(model_path, 'train_metrics_' + fold + '.npy'))
    val_res = np.load(os.path.join(model_path, 'val_metrics_' + fold + '.npy'))

    f1_val = val_res[:, 0]
    pr_val = val_res[:, 1]
    rec_val = val_res[:, 2]
    acc_val = val_res[:, 3]
    acc_one_val = val_res[:, 4]
    acc_zero_val = val_res[:, 5]
    loss_val = val_res[:, 6]

    f1_train = train_res[:, 0]
    pr_train = train_res[:, 1]
    rec_train = train_res[:, 2]
    acc_train = train_res[:, 3]
    acc_one_train = train_res[:, 4]
    acc_zero_train = train_res[:, 5]
    loss_train = train_res[:, 6]

    fig = plt.figure(figsize=[20, 10])

    ax1 = plt.subplot2grid((3,3), (0,0))
    block(ax1, f1_train, f1_val, 'F1 score, weighted', 'F1 score', 'Epoch')

    ax2 = plt.subplot2grid((3,3), (0,1))
    block(ax2, pr_train, pr_val, 'Precision, weighted', 'Precision', 'Epoch')

    ax3 = plt.subplot2grid((3,3), (0,2))
    block(ax3, rec_train, rec_val, 'Recall, weighted', 'Recall', 'Epoch')

    ax4 = plt.subplot2grid((3,3), (1,0))
    block(ax4, acc_train, acc_val, 'Accuracy, total', 'Accuracy', 'Epoch')

    ax5 = plt.subplot2grid((3,3), (1,1))
    block(ax5, acc_one_train, acc_one_val, 'Accuracy, cancer', 'Accuracy', 'Epoch')

    ax6 = plt.subplot2grid((3,3), (1,2))
    block(ax6, acc_zero_train, acc_zero_val, 'Accuracy, normal', 'Accuracy', 'Epoch')

    ax7 = plt.subplot2grid((3,3), (2,0), colspan=3)
    block(ax7, loss_train, loss_val, 'Loss, binary cross entropy', 'Loss', 'Epoch')

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.close(fig)

    if save:
        if save_dir is None:
            save_dir = model_path
        fig.savefig(os.path.join(save_dir, fold + save_format), bbox_inches='tight')
