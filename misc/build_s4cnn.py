from __future__ import division, print_function, absolute_import
import argparse
import os
import shutil
import sys
import time

import numpy as np
import tensorflow as tf

from six.moves import xrange    # pylint: disable=redefined-builtin
from datetime import datetime
from skimage.io import imread, imsave
from utils import make_dirs, manage_directories, ema, get_batches, get_data_files 

#from discrete_models import *
from so3_model import *
from modelnet_dataloader import ModelNetLoader

import npytar

##### Training #####
def train(args):
    print('...Building inputs')
    tf.reset_default_graph()

    print('...Connecting data io and preprocessing')
    k = 5
    train_inputs = tf.placeholder(tf.float32, [1,k,k,k,1], name="train_inputs")
    train_targets = tf.placeholder(tf.int32, [1,], name="train_targets")
    
    # Outputs
    print('...Constructing model')
    with tf.get_default_graph().as_default(): 
        with tf.variable_scope("model"):
            # TODO: move this
            trsh = train_inputs.get_shape().as_list()
            train_inputs = tf.reshape(train_inputs, [trsh[0], trsh[1], trsh[2], -1])
            angles = 2.*np.pi*tf.random_uniform([trsh[0],])
            train_inputs = tf.contrib.image.rotate(train_inputs, angles, interpolation="bilinear")
            train_inputs = tf.reshape(train_inputs, trsh)

            model = DiscreteModel(train_inputs, train_targets, args.n_classes, args, True)
            train_logits = model.pred_logits

    # Enter training loop: inputs to the graph are in capitals
    X = np.random.randn(1,k,k,k,1)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        out = sess.run(train_logits, feed_dict={train_inputs: X})
        out1 = out[0,2,2,2,0,:]
        print(out1)

        X = np.rot90(X, 3, (1,3))
        X = np.rot90(X, 3, (1,2))
        out = sess.run(train_logits, feed_dict={train_inputs: X})
        out2 = out[0,2,2,2,0,:]
        print(out2)

        P = []
        for i in range(24):
            val = (np.abs(out2 - out1[i]) < 1e-4)*1
            print(val)
            P.append(np.argmax(val))
        print(np.stack(P,0))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="minibatch size", type=int, default=64)
    parser.add_argument("--height", help="input image height", type=int, default=32)
    parser.add_argument("--width", help="input image width", type=int, default=32)
    parser.add_argument("--n_channels", help="number of input image channels", type=int, default=1)
    parser.add_argument("--n_channels_out", help="number of channel in first layer", type=int, default=32)
    parser.add_argument("--kernel_size", help="number of channel in first layer", type=int, default=3)
    parser.add_argument("--n_classes", help="number of output classes", type=int, default=10)
    parser.add_argument("--n_bases", help="number of bases", type=int, default=4)
    parser.add_argument("--frame_dim", help="number of dims per frame", type=int, default=8)

    parser.add_argument("--n_epochs", help="number of minibatches to pass", type=int, default=10)
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=1e-3)
    parser.add_argument("--learning_rate_step", help="interval to divide learning rate by 10", type=int, default=5)
    parser.add_argument("--momentum", help="momentum rate for stochastic gradient descent", type=float, default=0.9)

    parser.add_argument("--preprocess", help="whether to preprocess images", type=bool, default=True)
    parser.add_argument("--min_after_dequeue", help="minimum number of images to keep in RAM", type=int, default=500)

    parser.add_argument("--train_file", help="directory of training addresses", default="./shapenet10_train.tar")
    parser.add_argument("--valid_file", help="directory of validation addresses", default="./shapenet10_test.tar")
    parser.add_argument("--save_dir", help="directory to save results", default="./models/so3Feb26_0/checkpoints")
    parser.add_argument("--log_dir", help="directory to save results", default="./models/so3Feb26_0/logs")

    parser.add_argument("--path_increment", help="whether to automatically create new incremented directory", action="store_true")
    parser.add_argument("--save_interval", help="number of iterations between saving model", type=int, default=500)
    parser.add_argument("--delete_existing", help="delete existing models and logs in same folders", action="store_true")
    train(parser.parse_args())


