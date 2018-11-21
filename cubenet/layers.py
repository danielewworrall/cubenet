"""Models for the RFNN experiments"""
import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf


class Layers(object):
    def __init__(self, group):
        print(group)

        if group == "V":
            from V_group import V_group
            self.group = V_group()
            self.group_dim = self.group.group_dim
        elif group == "S4":
            from S4_group import S4_group
            self.group = S4_group()
            self.group_dim = self.group.group_dim
        elif group == "T4":
            from T4_group import T4_group
            self.group = T4_group()
            self.group_dim = self.group.group_dim
        else:
            print("Group is not recognized")
            sys.exit(-1)
            

        # Constants
        self.cayley = self.group.cayleytable


    def get_kernel(self, name, shape, factor=2.0, trainable=True):
        init = tf.contrib.layers.variance_scaling_initializer(factor=factor)
        return tf.get_variable(name, shape, initializer=init, trainable=trainable)


    def conv(self, x, kernel_size, n_out, strides=1, padding="SAME"):
        """A basic 3D convolution"""
        with tf.variable_scope("conv"):
            n_in = x.get_shape().as_list()[-1]
            W = self.get_kernel('W', [kernel_size,kernel_size,kernel_size,n_in,n_out])
            return tf.nn.conv3d(x, W, (1,strides,strides,strides,1), padding)


    def conv_block(self, x, kernel_size, n_out, is_training, use_bn=True, strides=1,
                   padding="SAME", fnc=tf.nn.relu, name="conv_block"):
        """Convolution with batch normalization/bias and nonlinearity"""
        with tf.variable_scope(name):
            y = self.conv(x, kernel_size, n_out, strides=strides, padding=padding)
            beta_init = tf.constant_initializer(0.01)
            if use_bn:
                return fnc(tf.layers.batch_normalization(y, training=is_training,
                           beta_initializer=beta_init))
            else:
                bias = tf.get_variable("bias", [n_out], initializer=beta_init)
                return fnc(tf.nn.bias_add(y, bias))


    def Gconv_block(self, x, kernel_size, n_out, is_training, use_bn=True, strides=1,
            padding="SAME", fnc=tf.nn.relu, name="Gconv_block", drop_sigma=0.1):
        """Convolution with batch normalization/bias and nonlinearity"""
        with tf.variable_scope(name):
            y = self.Gconv(x, kernel_size, n_out, is_training, strides=strides, padding=padding, drop_sigma=drop_sigma)
            beta_init = tf.constant_initializer(0.01)
            y = tf.transpose(y, perm=[0,1,2,3,5,4])
            ysh = y.get_shape().as_list()
            if use_bn:
                y = tf.layers.batch_normalization(y, training=is_training, beta_initializer=beta_init)
            else:
                bias = tf.get_variable("bias", [n_out], initializer=beta_init)
                y = tf.nn.bias_add(y, bias)
            return tf.transpose(fnc(y), perm=[0,1,2,3,5,4])


    def Gconv(self, x, kernel_size, n_out, is_training, strides=1, padding="SAME", drop_sigma=0.1):
        """Perform a discretized convolution on SO(3)

        Args:
            x: [batch_size, height, width, n_in, group_dim/1]
            kernel_size: int for the spatial size of the kernel
            n_out: int for number of output channels
            strides: int for spatial stride length
            padding: "valid" or "same" padding
        Returns:
            [batch_size, new_height, new_width, new_depth, n_out, group_dim] tensor in G
        """
        batch_size = tf.shape(x)[0]
        with tf.variable_scope('Gconv'):
            xsh = x.get_shape().as_list()
            init = tf.variance_scaling_initializer()
            # W is the base filter. We rotate it 4 times for a p4 convolution over
            # R^2. For a p4 convolution over p4, we rotate it, and then shift up by
            # one dimension in the channels.
            W = self.get_kernel("W", [kernel_size, kernel_size, kernel_size, xsh[4]*xsh[5]*n_out])
            WN = self.group.get_Grotations(W)
            WN = tf.stack(WN, -1)
            # Reshape and rotate the io filters 4 times. Each input-output pair is
            # rotated and stacked into a much bigger kernel
            xN = tf.reshape(x, [batch_size, xsh[1], xsh[2], xsh[3], xsh[4]*xsh[5]])
            if xsh[-1] == 1:
                # A convolution on R^2 is just standard convolution with 3 extra 
                # output channels for eacn rotation of the filters
                WN = tf.reshape(WN, [kernel_size, kernel_size, kernel_size, xsh[4], -1])
            elif xsh[-1] == self.group_dim:
                # A convolution on p4 is different to convolution on R^2. For each
                # dimension of the group output, we need to both rotate the filters
                # and circularly shift them in the input-group dimension. In a
                # sense, we have to spiral the filters
                WN = tf.reshape(WN, [kernel_size, kernel_size, kernel_size, xsh[4], self.group_dim, n_out, self.group_dim])
                # [kernel_size, kernel_size, kernel_size, n_in, 4, n_out, 4]
                # Shift over axis 4
                WN_shifted = self.group.G_permutation(WN)
                WN = tf.stack(WN_shifted, -1)
                # Shift over axis 6
                # Stack the shifted tensors and reshape to 4D kernel
                WN = tf.reshape(WN, [kernel_size, kernel_size, kernel_size, xsh[4]*self.group_dim, n_out*self.group_dim])

            # Convolve
            # Gaussian dropout on the weights
            WN *= (1 + drop_sigma*tf.to_float(is_training)*tf.random_normal(WN.get_shape()))

            if not (isinstance(strides, tuple) or isinstance(strides, list)):
                strides = (1,strides,strides,strides,1)
            if padding == 'REFLECT':
                padding = 'VALID'
                pad = WN.get_shape().as_list()[2] // 2
                xN = tf.pad(xN, [[0,0],[pad,pad],[pad,pad],[pad,pad],[0,0]], mode='REFLECT') 

            yN = tf.nn.conv3d(xN, WN, strides, padding)
            ysh = yN.get_shape().as_list()
            y = tf.reshape(yN, [batch_size, ysh[1], ysh[2], ysh[3], n_out, self.group_dim])
        return y


    def Gres_block(self, x, kernel_size, n_out, is_training, use_bn=True,
                   strides=1, padding="SAME", fnc=tf.nn.relu, drop_sigma=0.1,  name="Gres_block"):
        """Residual block style 3D group convolution
        
        Args:
            x: [batch_size, height, width, n_in, group_dim/1]
            kernel_size: int for the spatial size of the kernel
            n_out: int for number of output channels
            strides: int for spatial stride length
            padding: "valid" or "same" padding
        Returns:
            [batch_size, new_height, new_width, new_depth, n_out, group_dim] tensor in G
        """
        with tf.variable_scope(name):
            with tf.variable_scope("residual_connection"):
                # Begin residual connection
                y = self.Gconv_block(x, kernel_size, n_out, is_training, use_bn=use_bn, strides=strides, 
                                     padding=padding, fnc=fnc, drop_sigma=drop_sigma, name="Gconv_blocka")
                y = self.Gconv_block(y, kernel_size, n_out, is_training, use_bn=use_bn, drop_sigma=drop_sigma,
                                     fnc=tf.identity, name="Gconv_blockb")

            with tf.name_scope("shortcut_connection"):
                # Recombine with shortcut
                # a) resize and pad input if necessary
                xsh = tf.shape(x)
                ysh = tf.shape(y)
                xksize = (1,kernel_size,kernel_size,kernel_size,1)
                xstrides = (1,strides,strides,strides,1)
                x = tf.reshape(x, tf.concat([xsh[:4],[-1,]], 0))
                x = tf.nn.avg_pool3d(x, xksize, xstrides, "SAME")
                x = tf.reshape(x, tf.concat([ysh[:4],[-1,self.group_dim]], 0))
                
                diff = n_out - x.get_shape().as_list()[-2]
                paddings = tf.constant([[0,0],[0,0],[0,0],[0,0],[0,diff],[0,0]])
                x = tf.pad(x, paddings)
            
            with tf.name_scope("combinator"):
                # b) recombine
                #return fnc(x+y)
                return x+y
