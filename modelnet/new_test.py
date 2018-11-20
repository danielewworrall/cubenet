
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

#from V_model import *
from architectures import GVGG
from modelnet_loader import ModelNetLoader


##### Training #####
def test(args):
    print('...Building inputs')
    tf.reset_default_graph()

    print('...Connecting data io and preprocessing')
    n = 1
    test_inputs = tf.placeholder(tf.float32, [n*args.batch_size,32,32,32,1], name="test_inputs")
    test_targets = tf.placeholder(tf.int32, [n*args.batch_size,], name="test_targets")
    
    # Outputs
    print('...Constructing model')
    with tf.get_default_graph().as_default(): 
        with tf.variable_scope("model", reuse=False):
            model = GVGG(args, test_inputs, False)
            test_logits = model.pred_logits
            test_logits = tf.nn.sigmoid(test_logits)

            # Prediction loss
            print("...Building metrics")
            # If we rotation average, then we need to play about with the label
            # and prediction shapes
            """
            if args.group_rotations:
                test_logits = tf.reshape(test_logits, [-1, 12, args.n_classes])
                test_logits = tf.reduce_mean(test_logits, 1)
                # Targets
                targets = tf.reshape(test_targets, [-1, 12])[:,0]
            else:
                targets = test_targets
            preds = tf.to_int32(tf.argmax(test_logits, 1))
            test_accuracy = tf.contrib.metrics.accuracy(preds, targets)
            """
    
    # The data loader
    test_data = ModelNetLoader(args.test_folder)

    with tf.Session() as sess:
        # Load pretrained model, ignoring final layer
        print('...Restore variables')
        tf.global_variables_initializer().run()
        restorer = tf.train.Saver()
        model_path = tf.train.latest_checkpoint(args.save_dir)
        restorer.restore(sess, model_path)

        print("...Testing")
        test_accuracy_accumulator = 0.
        val_counter = 0
        iterator = test_data.iterator(args.batch_size, group_rotations=args.group_rotations)
        total_acc = 0.
        tcounter = 0.
        my_dict = {}

        for VALID_INPUTS, VALID_TARGETS, VALID_NAMES in iterator:
            VALID_INPUTS = np.concatenate(n*[VALID_INPUTS,], 0)
            VALID_TARGETS = np.concatenate(n*[VALID_TARGETS,], 0)
            mb_size = VALID_INPUTS.shape[0]
            val_counter += mb_size
            #sys.stdout.write("# Augmentations: {} \t# Examples {}\r".format(val_counter, val_counter/(1+11*args.group_rotations)))
            #sys.stdout.flush()            

            feed_dict = {test_inputs: VALID_INPUTS, test_targets: VALID_TARGETS}
            #acc, tl = sess.run([test_accuracy, test_logits], feed_dict=feed_dict)
            tl = sess.run(test_logits, feed_dict=feed_dict)
            #tl = np.mean(tl, 0)
            for name in VALID_NAMES:
                my_dict[name] = tl
            with open("./VMar8_3.pkl", 'wb') as fp:
                pkl.dump(my_dict, fp, pkl.HIGHEST_PROTOCOL)

            tl = np.reshape(tl, [12,n,-1])
            tl = np.mean(tl, 1)
            tl = np.mean(tl, 0)
            pd = np.argmax(tl, 0)
            
            total_acc += np.sum(np.equal(pd, VALID_TARGETS[0]))
            tcounter += 1 #pd.shape[0]
            #sys.stdout.write("{} ".format(total_acc / tcounter))
            print("# Augmentations: {} \t# Examples {}".format(total_acc / tcounter, tcounter))
            #test_accuracy_accumulator += acc*mb_size
            

        #current_test_accuracy = test_accuracy_accumulator / val_counter

        #print()
        #print("Test accuracy: {:04f}".format(current_test_accuracy))
        #print()


    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="minibatch size", type=int, default=12)
    parser.add_argument("--height", help="input image height", type=int, default=32)
    parser.add_argument("--width", help="input image width", type=int, default=32)
    parser.add_argument("--n_channels", help="number of input image channels", type=int, default=9*4)
    parser.add_argument("--kernel_size", help="number of channel in first layer", type=int, default=3)
    parser.add_argument("--first_kernel_size", help="number of channel in first layer", type=int, default=5)
    parser.add_argument("--n_classes", help="number of output classes", type=int, default=10)

    parser.add_argument("--group", help='group', type=str, default='V')
    parser.add_argument("--drop_sigma", help='dropout rate', type=float, default=0.1)
    parser.add_argument("--group_rotations", help="whether to rotation average", type=bool, default=True)
    parser.add_argument("--preprocess", help="whether to preprocess images", type=bool, default=True)
    parser.add_argument("--min_after_dequeue", help="minimum number of images to keep in RAM", type=int, default=500)

    parser.add_argument("--test_folder", help="directory of testation addresses", default="./data/modelnet10_test")
    parser.add_argument("--save_dir", help="directory to save results", default="../models2/VMar8_2/checkpoints")

    test(parser.parse_args())
