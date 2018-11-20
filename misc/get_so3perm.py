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


def get_s4mat():
    Z = []
    for i in range(4):
        # Z_4 rotation about Y
        # S^2 rotation
        for j in range(4):
            z = get_rotmat(i,j,0)
            Z.append(z)
        # Residual pole rotations
        Z.append(get_rotmat(i,0,1))
        Z.append(get_rotmat(i,0,3))
    return Z


def get_rotmat(x,y,z):
    c = [1.,0.,-1.,0.]
    s = [0.,1.,0.,-1]

    Rx = np.asarray([[c[x],     -s[x],  0.],
                     [s[x],     c[x],   0.],
                     [0.,       0.,     1.]])
    Ry = np.asarray([[c[y],     0.,     s[y]],
                     [0.,       1.,     0.],
                     [-s[y],    0.,     c[y]]])
    Rz = np.asarray([[1.,       0.,     0.],
                     [0.,       c[z],   -s[z]],
                     [0.,       s[z],   c[z]]])
    return Rz @ Ry @ Rx


def main():
    Z = get_s4mat()
    P = []
    for y in Z:
        for z in Z:
            r = z @ y
            for i, el in enumerate(Z):
                if np.sum(np.square(el - r)) < 1e-6:
                    P.append(i)
    P = np.stack(P)
    P = np.reshape(P, [24,24])
    print(P)
    
if __name__ == "__main__":
    main()

















































