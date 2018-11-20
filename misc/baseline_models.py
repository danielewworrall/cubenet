"""Models for the RFNN experiments"""
import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf

#from ops_layer import get_steerable_basis


class DiscreteModel(object):
    def __init__(self, images, labels, n_classes, args, is_training):
        # Constants
        self.batch_size = images.get_shape().as_list()[0]
        self.color_chn = images.get_shape().as_list()[3]

        # Placeholders
        self.is_training = tf.placeholder(tf.bool, [], name='is_training')
        self.learning_rate = args.learning_rate

        # Inputs
        self.images = images
        self.labels = labels 
        self.n_classes = n_classes
        self.fnc = tf.nn.relu
        self.ks = args.kernel_size
        self.frame_dim = args.frame_dim
        self.nc = args.n_channels_out

        # model predictions
        self.pred_logits, self.acts = self.get_pred(self.images, is_training)

        # Prediction loss
        self.loss = self.get_loss()
        preds = tf.to_int32(tf.argmax(self.pred_logits, 1))
        self.accuracy = tf.contrib.metrics.accuracy(preds, self.labels)
    

    def get_pred(self, x, is_training, reuse=False):
        acts = []
        with tf.variable_scope('prediction', reuse=reuse) as scope:
            init = tf.contrib.layers.variance_scaling_initializer()
            use_bn = True
            nc = self.nc

            x = conv_block(x, self.ks, nc, is_training, use_bn=use_bn, name="input_block")
            print(x)

            for i in range(4):
                with tf.variable_scope("VRN_{}".format(i)):
                    x = VRN_block(x, self.ks, nc, is_training, use_bn=use_bn, name="VRN_1")
                    x = VRN_block(x, self.ks, nc, is_training, use_bn=use_bn, name="VRN_2")
                    x = VRN_block(x, self.ks, nc, is_training, use_bn=use_bn, name="VRN_3")
                    nc = nc*2
                    x = vox_downsample(x, is_training, use_bn=use_bn, strides=2, name="vox_down")
                    print(x)
            x = conv_block(x, self.ks, nc, is_training, use_bn=use_bn, strides=1, name="output_block")
            print(x)

            keep_prob = 1. - 0.5*tf.to_float(is_training)
            # Cyclic pool (mean)
            #x = pNavg_pool(x, self.N, ksize=2, stride=2)
            x = tf.reduce_mean(x, [1,2,3])
            #x = tf.reduce_mean(x, [1,2,4])
            x = tf.reshape(x, [self.batch_size,1,1,1,-1])

            # Fully connected layers
            #x = tf.nn.dropout(x, keep_prob)
            x = conv_block(x, 1, 512, is_training, use_bn=False, name="fc1")

            x = tf.nn.dropout(x, keep_prob)
            with tf.variable_scope("logits"):
                pred_logits = conv_block(x, 1, self.n_classes, is_training, use_bn=False, fnc=tf.identity)
                return tf.squeeze(pred_logits), acts


    def get_loss(self):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.pred_logits)
        return tf.reduce_mean(loss, name='cross_entropy')


    def train_step(self, global_step):
        minimize_ops = []

        print('variables:')
        for var in tf.trainable_variables():
            print("\t{}".format(var))

        # training step for tied kernels
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_step_op = optimizer.minimize(self.loss, global_step=global_step)
        return train_step_op

    
##### Helpful functions #####
# Here we provide two flavors of convolution, standard and structured receptive field
## Conv
def get_kernel(name, shape, factor=2.0, trainable=True):
    init = tf.contrib.layers.variance_scaling_initializer(factor=factor)
    return tf.get_variable(name, shape, initializer=init, trainable=trainable)


def conv(x, kernel_size, n_out, strides=1, padding="SAME"):
    """A basic 2D convolution"""
    with tf.variable_scope("conv"):
        init = tf.contrib.layers.variance_scaling_initializer()
        n_in = x.get_shape().as_list()[-1]
        shape = [kernel_size,kernel_size,kernel_size,n_in,n_out]
        W = tf.get_variable('W', shape, initializer=init)
        return tf.nn.conv3d(x, W, (1,strides,strides,strides,1), padding)


def conv_block(x, kernel_size, n_out, is_training, use_bn=True, strides=1,
               padding="SAME", fnc=tf.nn.relu, name="conv_block"):
    """Convolution with batch normalization/bias and nonlinearity"""
    with tf.variable_scope(name):
        y = conv(x, kernel_size, n_out, strides=strides, padding=padding)
        beta_init = tf.constant_initializer(0.01)
        if use_bn:
            return fnc(tf.layers.batch_normalization(y, training=is_training,
                       beta_initializer=beta_init))
        else:
            bias = tf.get_variable("bias", [n_out], initializer=beta_init)
            return fnc(tf.nn.bias_add(y, bias))


def resnet_block(x, kernel_size, n_out, is_training, use_bn=True, strides=1,
                 padding="SAME", fnc=tf.nn.relu, name="resnet_block"):
    """Residual network block"""
    with tf.variable_scope(name):
        # Subsample
        x_skip = tf.nn.avg_pool3d(x, (1,1,1,1,1), (1,strides,strides,strides,1), "SAME")

        # Convolutional block
        x = conv_block(x, kernel_size, n_out, is_training, use_bn=use_bn, 
                       strides=strides, fnc=fnc, name="conv_1")
        x = conv_block(x, kernel_size, n_out, is_training, use_bn=use_bn, 
                       strides=1, fnc=tf.identity, name="conv_2")

        # The residual connection
        pad = x.get_shape().as_list()[-1] - x_skip.get_shape().as_list()[-1]
        paddings = [[0,0],[0,0],[0,0],[0,0],[0,pad]]
        x_skip = tf.pad(x_skip, paddings, "CONSTANT")

        return fnc(x+x_skip) 


def vox_downsample(x, is_training, use_bn=True, strides=2, fnc=tf.nn.relu, name="vox"):
    with tf.variable_scope(name):
        beta_init = tf.constant_initializer(0.01)
        n_out = x.get_shape().as_list()[-1]
        n_out = int(n_out/2)
        ymax = conv_block(x, 3, n_out, is_training, use_bn=False, strides=1,
                          padding="SAME", fnc=tf.identity, name='max')
        ymax = tf.nn.max_pool3d(ymax, (1,2,2,2,1), (1,strides,strides,strides,1), "SAME")
        ymax = tf.layers.batch_normalization(ymax, training=is_training, beta_initializer=beta_init)

        yavg = conv_block(x, 3, n_out, is_training, use_bn=False, strides=1,
                          padding="SAME", fnc=tf.identity, name='avg')
        yavg = tf.nn.avg_pool3d(yavg, (1,2,2,2,1), (1,strides,strides,strides,1), "SAME")
        yavg = tf.layers.batch_normalization(yavg, training=is_training, beta_initializer=beta_init)

        y3x3 = conv_block(x, 3, n_out, is_training, use_bn=use_bn, strides=strides,
                          padding="SAME", fnc=tf.identity, name='3x3')
        y1x1 = conv_block(x, 3, n_out, is_training, use_bn=use_bn, strides=strides,
                          padding="SAME", fnc=tf.identity, name='1x1')
        
        y = tf.concat([ymax, yavg, y3x3, y1x1], -1)
        y = tf.layers.batch_normalization(y, training=is_training, beta_initializer=beta_init)
        return fnc(y) 


def VRN_block(x, kernel_size, n_out, is_training, use_bn=True, strides=1,
        padding="SAME", fnc=tf.nn.relu, name="resnet_block"):
    """Voxception residual network block"""
    with tf.variable_scope(name):
        beta_init = tf.constant_initializer(0.01)
        # Subsample
        x_skip = tf.nn.avg_pool3d(x, (1,1,1,1,1), (1,strides,strides,strides,1), "SAME")

        # 3-3 convolutional block
        with tf.variable_scope("branch_a"):
            y = conv_block(x, kernel_size, int(n_out/4), is_training, use_bn=use_bn, 
                           strides=strides, fnc=fnc, name="conv_3x3a")
            y = conv_block(y, kernel_size, int(n_out/2), is_training, use_bn=False, 
                           strides=1, fnc=tf.identity, name="conv_3x3b")

        # 1-3-1 convolutional block
        with tf.variable_scope("branch_b"):
            z = conv_block(x, 1, int(n_out/4), is_training, use_bn=use_bn, 
                           strides=strides, fnc=fnc, name="conv_1x1a")
            z = conv_block(z, kernel_size, int(n_out/4), is_training, use_bn=use_bn, 
                           strides=1, fnc=fnc, name="conv_3x3b")
            z = conv_block(z, 1, int(n_out/2), is_training, use_bn=False, 
                           strides=1, fnc=tf.identity, name="conv_1x1c")

        x = tf.concat([y,z], -1)

        # The residual connection
        pad = x.get_shape().as_list()[-1] - x_skip.get_shape().as_list()[-1]
        paddings = [[0,0],[0,0],[0,0],[0,0],[0,pad]]
        x_skip = tf.pad(x_skip, paddings, "CONSTANT")

        v = tf.layers.batch_normalization(x + x_skip, training=is_training, beta_initializer=beta_init)

        return fnc(v) 
    
