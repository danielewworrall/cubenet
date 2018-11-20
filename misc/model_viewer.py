import argparse
import os
import sys
import time

import numpy as np

import npytar

import importlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_batches(args, filename, augment_data=False):
    """Read contents of address file and store in lists"""
    reader = npytar.NpyTarReader(filename)

    images_list = []
    labels_list = []
    for i, (x, name) in enumerate(reader):
        # Data augmentation
        if augment_data == True:
            x = augment(x)
        images_list.append(x)
        labels_list.append(int(name.split(".")[0])-1)
        if (i+1) % args.batch_size == 0:
            images = np.stack(images_list, 0)
            labels = np.stack(labels_list, 0).astype(np.int32)
            images_list = []
            labels_list = []
            yield images[...,np.newaxis], labels


def augment(x):
    axes = [(0,1), (1,2), (2,0)]
    x = np.flip(x, 2)
    for i in range(3):
        # Flip
        #if np.random.rand() > 0.5:
        #    x = np.flip(x, i)
        
        # Shift
        max_shift = 3
        paddings = [(0,0),(0,0),(0,0)]
        paddings[i] = (max_shift, max_shift)
        x = np.pad(x, paddings, mode="constant")
        shift = int(np.floor(np.random.rand()*(2*max_shift+1)))
        # Slice object
        slice_object = list(np.s_[:,:,:])
        slice_object[i] = np.s_[shift:shift+32]
        slice_object = tuple(slice_object)
        x = x[slice_object]

        # Rotate
        #k = int(np.floor(np.random.rand()*4))
        #x = np.rot90(x, k=k, axes=axes[i])
    return x


def view(args):
    fig = plt.figure(1)
    plt.ion()
    plt.show()
    for images, labels in get_batches(args, "./shapenet10_train.tar", augment_data=False):
        voxels = images[0,...,0]
        print(voxels.shape)
        voxels = voxels[::-1,:,:]

        # and plot everything
        plt.cla()
        ax = fig.gca(projection='3d')
        ax.voxels(voxels, edgecolor='k')

        plt.draw()
        input()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="minibatch size", type=int, default=1)
    view(parser.parse_args())
