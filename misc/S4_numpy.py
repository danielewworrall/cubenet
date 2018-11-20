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


def display(X):
    fig = plt.figure(1, figsize=(12,4))
    for i in range(24):
        x = X[i]
        ax = fig.add_subplot(4,6,i+1,projection='3d')
        alpha = 0.9*np.ones(x.shape)
        colors = np.stack([x,x,x,alpha], -1)
        ax.voxels(x, facecolors=colors, edgecolor='k')
        plt.axis("square")
    plt.show()


def get_so3(x):
    Z = []
    for i in range(4):
    # Z_4 rotation about Y
        y = rotate(x, 2, i)
        # S^2 rotation
        for j in range(4):
            z = rotate(y, 0, j)
            Z.append(z)
        # Residual pole rotations
        Z.append(rotate(y, 1, 1))
        Z.append(rotate(y, 1, 3))
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
