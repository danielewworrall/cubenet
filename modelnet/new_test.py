
import argparse
import os
import shutil
import sys
import time
sys.path.append('..')
sys.path.append('../cubenet')

import pickle as pkl
import numpy as np
import tensorflow as tf

from skimage.io import imread, imsave
from utils import make_dirs, manage_directories 

from architectures import GVGG
from dataloader import DataLoader


##### Training #####
def test(args):
    print('...Building inputs')
    tf.reset_default_graph()

    print('...Connecting data io and preprocessing')
    with tf.device("/cpu:0"):
        with tf.name_scope("IO"):
            test_data = DataLoader(args.test_file, 'test', args.batch_size,
                                    args.height, args.jitter, shuffle=False)
            args.n_classes = test_data.n_classes
            args.data_size = test_data.data_size
            print("Found {} test examples".format(args.data_size))

            test_iterator = test_data.data.make_initializable_iterator()
            test_inputs, test_targets = test_iterator.get_next()
            test_inputs.set_shape([args.batch_size, args.height, args.width, args.depth, 1])
            test_init_op = test_iterator.make_initializer(test_data.data)
    
    # Outputs
    print('...Constructing model')
    with tf.get_default_graph().as_default(): 
        with tf.variable_scope("model", reuse=False):
            model = GVGG(test_inputs, False, args)
            test_logits = model.pred_logits
            test_logits = tf.nn.sigmoid(test_logits)

            # Prediction loss
            print("...Building metrics")
            # If we rotation average, then we need to play about with the label
            # and prediction shapes
            preds = tf.to_int32(tf.argmax(test_logits, 1))
            test_accuracy = tf.contrib.metrics.accuracy(preds, test_targets)
    
    with tf.Session() as sess:
        # Load pretrained model, ignoring final layer
        print('...Restore variables')
        tf.global_variables_initializer().run()
        restorer = tf.train.Saver()
        model_path = tf.train.latest_checkpoint(args.save_dir)
        restorer.restore(sess, model_path)

        accuracies = []
        print("...Testing")

        sess.run([test_init_op])
        for i in range(args.data_size // args.batch_size):
            accuracies.append(sess.run(test_accuracy))
            sys.stdout.write("[{} | {}] Running acc: {}\r".format(i*args.batch_size, args.data_size, np.mean(accuracies)))
            sys.stdout.flush()
            
        print()
        print("Test accuracy: {:04f}".format(np.mean(accuracies)))
        print()


    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="minibatch size", type=int, default=12)
    parser.add_argument("--height", help="input image height", type=int, default=32)
    parser.add_argument("--width", help="input image width", type=int, default=32)
    parser.add_argument("--depth", help="input image depth", type=int, default=32)
    parser.add_argument("--n_channels", help="number of input image channels", type=int, default=9*4)
    parser.add_argument("--kernel_size", help="number of channel in first layer", type=int, default=3)
    parser.add_argument("--first_kernel_size", help="number of channel in first layer", type=int, default=5)
    parser.add_argument("--n_classes", help="number of output classes", type=int, default=10)
    parser.add_argument("--jitter", help="amount of test time jitter", type=int, default=0)

    parser.add_argument("--group", help='group', type=str, default='V')
    parser.add_argument("--drop_sigma", help='dropout rate', type=float, default=0.1)
    parser.add_argument("--group_rotations", help="whether to rotation average", type=bool, default=True)
    parser.add_argument("--preprocess", help="whether to preprocess images", type=bool, default=True)
    parser.add_argument("--min_after_dequeue", help="minimum number of images to keep in RAM", type=int, default=500)

    parser.add_argument("--test_file", help="directory of test addresses", default="./addresses/modelnet10_test_addresses.txt")
    parser.add_argument("--save_dir", help="directory to save results", default="./models/model_0/checkpoints")

    test(parser.parse_args())
