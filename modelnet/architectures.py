"""Models for the RFNN experiments"""
import argparse
import os
import sys
import time
sys.path.append('../')
sys.path.append('../cubenet')

import numpy as np
import tensorflow as tf

from layers import Layers


class GVGG(Layers):
    def __init__(self, images, is_training, args):
        super().__init__(args.group)

        # Constants
        self.batch_size = tf.shape(images)[0]

        # Inputs
        self.images = images
        self.n_classes = args.n_classes
        self.ks = args.kernel_size
        self.ks1 = args.first_kernel_size
        self.nc = args.n_channels
        self.drop_sigma = args.drop_sigma

        # model predictions
        print("...Constructing network")
        self.pred_logits = self.get_pred(self.images, is_training)


    def get_pred(self, x, is_training): #, reuse=False):
        with tf.variable_scope('prediction') as scope: #, reuse=reuse) as scope:
            init = tf.contrib.layers.variance_scaling_initializer()
            use_bn = True
            nc = int(self.nc/self.group_dim)
            ds = self.drop_sigma

            x = tf.expand_dims(x, -1)
            x = self.Gconv_block(x, self.ks1, nc, is_training, use_bn=use_bn, drop_sigma=ds, name="Gconv_1a")
            x = self.Gconv_block(x, self.ks, nc, is_training, use_bn=use_bn, drop_sigma=ds, name="Gconv_1b")
            print(x)
            x = self.Gconv_block(x, self.ks, 2*nc, is_training, use_bn=use_bn, strides=2, drop_sigma=ds, name="Gconv_2a")
            x = self.Gconv_block(x, self.ks, 2*nc, is_training, use_bn=use_bn, strides=1, drop_sigma=ds, name="Gconv_2b")
            print(x)
            x = self.Gconv_block(x, self.ks, 4*nc, is_training, use_bn=use_bn, strides=2, drop_sigma=ds, name="Gconv_3a")
            x = self.Gconv_block(x, self.ks, 4*nc, is_training, use_bn=use_bn, strides=1, drop_sigma=ds, name="Gconv_3b")
            print(x)
            x = self.Gconv_block(x, self.ks, 8*nc, is_training, use_bn=use_bn, strides=2, drop_sigma=ds, name="Gconv_4a")
            x = self.Gconv_block(x, self.ks, 8*nc, is_training, use_bn=use_bn, strides=1, drop_sigma=ds, name="Gconv_4b")
            print(x)
            x = self.Gconv_block(x, self.ks, 16*nc, is_training, use_bn=use_bn, strides=2, drop_sigma=ds, name="Gconv_5a")
            x = self.Gconv_block(x, self.ks, 16*nc, is_training, use_bn=use_bn, strides=1, drop_sigma=ds, name="Gconv_5b")
            print(x)


            keep_prob = 1. - 0.5*tf.to_float(is_training)
            # Cyclic pool (mean)
            x = tf.reduce_mean(x, [1,2,3,5])
            x = tf.reshape(x, [self.batch_size,1,1,1,x.get_shape().as_list()[-1]])

            # Fully connected layers
            x = tf.nn.dropout(x, keep_prob)
            x = self.conv_block(x, 1, 512, is_training, use_bn=False, name="fc1")

            x = tf.nn.dropout(x, keep_prob)
            with tf.variable_scope("logits"):
                pred_logits = self.conv_block(x, 1, self.n_classes, is_training, use_bn=False, fnc=tf.identity)
                return tf.squeeze(pred_logits)


class GResnet(Layers):
    def __init__(self, images, is_training, args):
        super().__init__(args.group)

        # Constants
        self.batch_size = tf.shape(images)[0]

        # Inputs
        self.images = images
        self.n_classes = args.n_classes
        self.ks = args.kernel_size
        self.ks1 = args.first_kernel_size
        self.nc = args.n_channels
        self.drop_sigma = args.drop_sigma

        # model predictions
        print("...Constructing network")
        self.pred_logits = self.get_pred(self.images, is_training)


    def get_pred(self, x, is_training): #, reuse=False):
        with tf.variable_scope('prediction') as scope: #, reuse=reuse) as scope:
            init = tf.contrib.layers.variance_scaling_initializer()
            use_bn = True
            nc = int(self.nc/self.group_dim)
            ds = self.drop_sigma

            x = tf.expand_dims(x, -1)
            x = self.Gconv_block(x, self.ks1, nc, is_training, use_bn=use_bn, drop_sigma=ds, name="Gconv_0")
            print(x)

            x = self.Gres_block(x, self.ks, 2*nc, is_training, use_bn=use_bn, strides=2, drop_sigma=ds, name="Gres_1a")
            x = self.Gres_block(x, self.ks, 2*nc, is_training, use_bn=use_bn, drop_sigma=ds, name="Gres_1b")
            x = self.Gres_block(x, self.ks, 2*nc, is_training, use_bn=use_bn, drop_sigma=ds, name="Gres_1c")
            print(x)

            x = self.Gres_block(x, self.ks, 4*nc, is_training, use_bn=use_bn, strides=2, drop_sigma=ds, name="Gres_2a")
            x = self.Gres_block(x, self.ks, 4*nc, is_training, use_bn=use_bn, drop_sigma=ds, name="Gres_2b")
            x = self.Gres_block(x, self.ks, 4*nc, is_training, use_bn=use_bn, drop_sigma=ds, name="Gres_2c")
            print(x)

            x = self.Gres_block(x, self.ks, 8*nc, is_training, use_bn=use_bn, strides=2, drop_sigma=ds, name="Gres_3a")
            x = self.Gres_block(x, self.ks, 8*nc, is_training, use_bn=use_bn, drop_sigma=ds, name="Gres_3b")
            x = self.Gres_block(x, self.ks, 8*nc, is_training, use_bn=use_bn, drop_sigma=ds, name="Gres_3c")
            print(x)

            keep_prob = 1. - 0.5*tf.to_float(is_training)
            # Cyclic pool (mean)
            x = tf.reduce_mean(x, [1,2,3,5])
            x = tf.reshape(x, [self.batch_size,1,1,1,x.get_shape().as_list()[-1]])

            # Fully connected layers
            x = tf.nn.dropout(x, keep_prob)
            x = self.conv_block(x, 1, 512, is_training, use_bn=False, name="fc1")

            x = tf.nn.dropout(x, keep_prob)
            with tf.variable_scope("logits"):
                pred_logits = self.conv_block(x, 1, self.n_classes, is_training, use_bn=False, fnc=tf.identity)
                return tf.squeeze(pred_logits)


