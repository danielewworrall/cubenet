import os
import sys
import time

import numpy as np

import importlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rotate(x, axis, shift):
    """Rotate x shift times about axis (0,1,2)"""
    axes = [(0,1), (1,2), (0,2)]
    return np.rot90(x, k=shift, axes=axes[axis])


def r1(x):
    x = rotate(x, 0, 1)
    return rotate(x, 1, -1)

    
def r2(x):
    x = rotate(x, 0, 1)
    return rotate(x, 1, 1)


def r3(x):
    x = rotate(x, 0, 2)
    return x


def display(X):
    fig = plt.figure(1, figsize=(12,4))
    for i in range(12):
        x = X[i]
        ax = fig.add_subplot(4,3,i+1,projection='3d')
        alpha = 0.75*np.ones(x.shape)
        colors = np.stack([x,x,x,alpha], -1)
        ax.voxels(x, facecolors=colors, edgecolor='k')
        plt.axis("square")
    plt.show()


def get_so3(x):
    Z = []
    for i in range(3):
        y = x.copy()
        for __ in range(i):
            y = r1(y) 
        for j in range(3):
            z = y.copy()
            for __ in range(j):
                z = r2(z)
            Z.append(z)
    for i in range(3):
        z = r3(x)
        for __ in range(i):
            z = r2(z) 
        Z.append(z)
    return Z


def main():
    #x = np.random.rand(3,3,3)
    x = np.ones((3,3,3))
    x[0,1,2] = 0.1
    x[1,1,2] = 0.1
    Z = get_so3(x)

    display(Z)


if __name__ == "__main__":
    main()
