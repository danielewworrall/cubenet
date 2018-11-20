import os
import sys
import time

import numpy as np
import tensorflow as tf


class V_group(object):
    def __init__(self):
        self.group_dim = 4
        self.cayleytable = self.get_cayleytable()


    def get_cayleytable(self):
        print("...Computing Cayley table")
        cayley = np.asarray([[0,1,2,3],
                             [1,0,3,2],
                             [2,3,0,1],
                             [3,2,1,0]])
        return cayley        


    def get_Grotations(self, x):
        """Rotate the tensor x with all 4 Klein Vierergruppe rotations

        Args:
            x: [h,w,d,n_channels]
        Returns:
            list of 4 rotations of x [[h,w,d,n_channels],....]
        """
        xsh = x.get_shape().as_list()
        angles = [0.,np.pi]
        rx = []
        for i in range(2):
            # 2x 180. rotations about the z axis
            perm = [1,0,2,3]
            y = tf.transpose(x, perm=perm)
            y = tf.contrib.image.rotate(y, angles[i])
            y = tf.transpose(y, perm=perm)

            # 2x 180. rotations about another axis
            for j in range(2):
                perm = [2,1,0,3]
                z = tf.transpose(y, perm=perm)
                z = tf.contrib.image.rotate(z, angles[j])
                z = tf.transpose(z, perm=perm)
                rx.append(z)
        return rx


    def G_permutation(self, W):
        """Permute the outputs of the group convolution"""
        Wsh = W.get_shape().as_list()
        cayley = self.cayleytable
        U = []
        for i in range(4):
            perm_mat = self.get_permutation_matrix(cayley, i)
            w = W[:,:,:,:,:,:,i]
            w = tf.transpose(w, [0,1,2,3,5,4])
            w = tf.reshape(w, [-1, 4])
            w = w @ perm_mat
            w = tf.reshape(w, Wsh[:4]+[-1,4])
            U.append(tf.transpose(w, [0,1,2,3,5,4]))
        return U


    def get_permutation_matrix(self, perm, dim):
        ndim = perm.shape[0]
        mat = np.zeros((ndim, ndim))
        for j in range(ndim):
            mat[j,perm[j,dim]] = 1
        return mat
