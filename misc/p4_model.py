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
        self.nc = args.n_channels_out
        self.group_dim = 4

        # model predictions
        print("...Constructing network")
        self.pred_logits, self.acts = self.get_pred(self.images, is_training)

        # Prediction loss
        print("...Building loss")
        self.loss = self.get_loss()
        preds = tf.to_int32(tf.argmax(self.pred_logits, 1))
        self.accuracy = tf.contrib.metrics.accuracy(preds, self.labels)

    def get_pred(self, x, is_training, reuse=False):
        acts = []
        with tf.variable_scope('prediction', reuse=reuse) as scope:
            init = tf.contrib.layers.variance_scaling_initializer()
            use_bn = True
            nc = int(self.nc/self.group_dim)

            x = tf.expand_dims(x, -1)
            x = p4conv_block(x, 5, nc, is_training, use_bn=use_bn, name="so3_1a")
            x = p4conv_block(x, self.ks, nc, is_training, use_bn=use_bn, name="so3_1b")
            print(x)
            x = p4conv_block(x, self.ks, 2*nc, is_training, use_bn=use_bn, strides=2, name="so3_2a")
            x = p4conv_block(x, self.ks, 2*nc, is_training, use_bn=use_bn, strides=1, name="so3_2b")
            print(x)
            x = p4conv_block(x, self.ks, 4*nc, is_training, use_bn=use_bn, strides=2, name="so3_3a")
            x = p4conv_block(x, self.ks, 4*nc, is_training, use_bn=use_bn, strides=1, name="so3_3b")
            print(x)
            x = p4conv_block(x, self.ks, 8*nc, is_training, use_bn=use_bn, strides=2, name="so3_4a")
            x = p4conv_block(x, self.ks, 8*nc, is_training, use_bn=use_bn, strides=1, name="so3_4b")
            print(x)
            x = p4conv_block(x, self.ks, 16*nc, is_training, use_bn=use_bn, strides=2, name="so3_5a")
            x = p4conv_block(x, self.ks, 16*nc, is_training, use_bn=use_bn, strides=1, name="so3_5b")
            print(x)


            keep_prob = 1. - 0.5*tf.to_float(is_training)
            # Cyclic pool (mean)
            #x = pNavg_pool(x, group_dim, ksize=2, stride=2)
            x = tf.reduce_mean(x, [1,2,3,5])
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
    """A basic 3D convolution"""
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


def p4conv_block(x, kernel_size, n_out, is_training, use_bn=True, strides=1,
               padding="SAME", fnc=tf.nn.relu, name="p4conv_block"):
    """Convolution with batch normalization/bias and nonlinearity"""
    with tf.variable_scope(name):
        y = p4conv(x, kernel_size, n_out, strides=strides, padding=padding)
        beta_init = tf.constant_initializer(0.01)
        y = tf.transpose(y, perm=[0,1,2,3,5,4])
        ysh = y.get_shape().as_list()
        if use_bn:
            y = tf.layers.batch_normalization(y, training=is_training, beta_initializer=beta_init)
        else:
            bias = tf.get_variable("bias", [n_out], initializer=beta_init)
            y = tf.nn.bias_add(y, bias)
        return tf.transpose(fnc(y), perm=[0,1,2,3,5,4])


def p4conv(x, kernel_size, n_out, strides=1, padding="SAME"):
    """Perform a discretized convolution on SO(3)

    Args:
        x: [batch_size, height, width, n_in, group_dim/1]
        kernel_size: int for the spatial size of the kernel
        n_out: int for number of output channels
        strides: int for spatial stride length
        padding: "valid" or "same" padding
    Returns:
        [batch_size, new_height, new_width, new_depth, n_out, group_dim] tensor in p4
    """
    group_dim = 4
    with tf.variable_scope('pNconv'):
        xsh = x.get_shape().as_list()
        init = tf.variance_scaling_initializer()
        # W is the base filter. We rotate it 4 times for a p4 convolution over
        # R^2. For a p4 convolution over p4, we rotate it, and then shift up by
        # one dimension in the channels.
        W = get_kernel("W", [kernel_size, kernel_size, kernel_size*xsh[4]*xsh[5]*n_out])
        WN = []
        for i in range(group_dim):
            angle = 2.*np.pi*(i / group_dim)
            WN.append(tf.contrib.image.rotate(W, angle))
        WN = tf.stack(WN, -1)
        # Reshape and rotate the io filters 4 times. Each input-output pair is
        # rotated and stacked into a much bigger kernel
        xN = tf.reshape(x, [xsh[0], xsh[1], xsh[2], xsh[3], -1])
        if xsh[-1] == 1:
            # A convolution on R^2 is just standard convolution with 3 extra 
            # output channels for eacn rotation of the filters
            WN = tf.reshape(WN, [kernel_size, kernel_size, kernel_size, xsh[4], -1])
        elif xsh[-1] == group_dim:
            # A convolution on p4 is different to convolution on R^2. For each
            # dimension of the group output, we need to both rotate the filters
            # and circularly shift them in the input-group dimension. In a
            # sense, we have to spiral the filters
            WN = tf.reshape(WN, [kernel_size, kernel_size, kernel_size, xsh[4], group_dim, n_out, group_dim])
            print("WN: {}".format(WN))
            # [kernel_size, kernel_size, kernel_size, n_in, 4, n_out, 4]
            # Shift over axis 4
            WN_shifted = []
            for i in range(group_dim):
                WN_shifted.append(circular_shift(WN[:,:,:,:,:,:,i], i, 4))
            WN = tf.stack(WN_shifted, -1)
            # Shift over axis 6
            # Stack the shifted tensors and reshape to 4D kernel
            WN = tf.reshape(WN, [kernel_size, kernel_size, kernel_size, xsh[4]*group_dim, n_out*group_dim])

        # Convolve
        yN = tf.nn.conv3d(xN, WN, (1,strides,strides,strides,1), padding)
        ysh = yN.get_shape().as_list()
        y = tf.reshape(yN, [ysh[0], ysh[1], ysh[2], ysh[3], n_out, group_dim])
    return y


def circular_shift(x, shift, axis):
    """Shift x by shift along axis axis

    Args:
        x: input tensor
        shift: int shift amount
        axis: int axis for circular shift
    Returns:
        shifted tensor of shape x
    """
    if shift == 0:
        y = x
    else:
        xsh = x.get_shape().as_list()
        rank = len(xsh)
        dim = xsh[-2]
        assert axis >= 0, 'Require axis >= 0, but axis={}'.format(axis)
        assert axis < rank, 'Require axis < rank(x), but axis={} and rank={}'.format(axis, rank)

        # Split tensor into 2 parts and reorder
        # 1st half
        begin = np.zeros((rank), dtype=np.int32)
        size = xsh
        #size[axis] = rank-shift-1
        size[axis] = dim-shift
        before = tf.slice(x, tf.constant(begin), tf.constant(size))

        # 2nd half
        begin = np.zeros((rank), dtype=np.int32)
        #begin[axis] = rank-shift-1
        begin[axis] = dim-shift
        size = xsh
        size[axis] = shift
        after = tf.slice(x, tf.constant(begin), tf.constant(size))

        y = tf.concat([after, before], axis=axis)
    return y


