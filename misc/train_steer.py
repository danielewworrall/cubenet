from __future__ import division, print_function, absolute_import
import argparse
import os
import shutil
import sys
import time
sys.path.append('./architectures')

import numpy as np
import tensorflow as tf

from skimage.io import imread, imsave
from utils import make_dirs, manage_directories 

from V_model_steer import *
from dataloader import DataLoader


def get_learning_rate(base_learning_rate, learning_rate_step, global_step):
    schedule = np.power(0.2, np.floor(global_step / learning_rate_step))
    return base_learning_rate * schedule


##### Training #####
def train(args):
    print('Setting up')
    args = manage_directories(args)
    print('...Building inputs')
    tf.reset_default_graph()

    # Inputs to the model
    with tf.name_scope('IO'):
        learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
        global_step = tf.Variable(0, trainable=False, name='global_step')
        valid_accuracy_ph = tf.placeholder(tf.float32, [], name='valid_accuracy_ph')
    

    print('...Connecting data io and preprocessing')
    with tf.device("/cpu:0"):
        with tf.name_scope("IO"):
            train_data = DataLoader(args.train_file, 'train', args.batch_size,
                                    args.n_classes, args.height, args.jitter, shuffle=True)
            valid_data = DataLoader(args.valid_file, 'test', args.batch_size,
                                    args.n_classes, args.height, args.jitter, shuffle=False)

            train_iterator = train_data.data.make_initializable_iterator()
            valid_iterator = valid_data.data.make_initializable_iterator()

            train_inputs, train_targets = train_iterator.get_next()
            train_inputs.set_shape([args.batch_size, args.height, args.width, args.depth, args.n_channels])
            valid_inputs, valid_targets = valid_iterator.get_next()
            valid_inputs.set_shape([args.batch_size, args.height, args.width, args.depth, args.n_channels])

            train_init_op = train_iterator.make_initializer(train_data.data)
            valid_init_op = valid_iterator.make_initializer(valid_data.data)


    # Outputs
    print('...Constructing model')
    with tf.get_default_graph().as_default(): 
        with tf.variable_scope("model"):
            model = DiscreteModel(train_inputs, args.n_classes, args, True)
            train_logits = model.pred_logits
            train_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_targets, logits=train_logits)
            train_loss = tf.reduce_mean(train_loss, -1)

            preds = tf.to_int32(tf.argmax(train_logits, 1))
            train_accuracy = tf.contrib.metrics.accuracy(preds, train_targets)

        with tf.variable_scope("model", reuse=True):
            model = DiscreteModel(valid_inputs, args.n_classes, args, False)
            valid_logits = model.pred_logits
            valid_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=valid_targets, logits=valid_logits)
            valid_loss = tf.reduce_mean(valid_loss, -1)

            preds = tf.to_int32(tf.argmax(valid_logits, 1))
            valid_accuracy = tf.contrib.metrics.accuracy(preds, valid_targets)

    # Compute train ops
    print('Variables:')
    for var in tf.trainable_variables():
        print("\t{}".format(var))

    with tf.name_scope("Optimizer"):
        # Instantiate optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Run batch norm updates as well as gradient descent step
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(train_loss, global_step=global_step)

    # Summaries
    with tf.name_scope('Summaries'):
        train_summary = []
        valid_summary = []

        train_summary.append(tf.summary.scalar('Learning_rate', learning_rate))
        train_summary.append(tf.summary.scalar('Train_loss', train_loss))
        train_summary.append(tf.summary.scalar('Train_accuracy', train_accuracy))
        valid_summary.append(tf.summary.scalar('Valid_accuracy', valid_accuracy_ph))


    train_batches_per_epoch = int(np.floor(train_data.data_size / args.batch_size))
    valid_batches_per_epoch = int(np.floor(valid_data.data_size / args.batch_size))
    print(train_batches_per_epoch, valid_batches_per_epoch)

    # Enter training loop: inputs to the graph are in capitals
    with tf.Session() as sess:
        # Merge all the summaries
        train_summary = tf.summary.merge(train_summary, name='train_summary')
        valid_summary = tf.summary.merge(valid_summary, name='valid_summary')
        train_writer = tf.summary.FileWriter(args.log_dir + '/train', sess.graph)
        valid_writer = tf.summary.FileWriter(args.log_dir + '/valid')

        # Load pretrained model, ignoring final layer
        print('...Initializing new variables')
        tf.global_variables_initializer().run()


        """
        # Finetuning
        restorables = [var for var in tf.trainable_variables() if "logit" not in var.name]
        restorer = tf.train.Saver(var_list=restorables)
        load_path = tf.train.latest_checkpoint('./models/VMar1_1/checkpoints')
        restorer.restore(sess, load_path)    
        """

        saver = tf.train.Saver()

        print('Training')
        start = time.time()
        current_valid_accuracy = 0
        for i in range(args.n_epochs):
            sess.run([train_init_op, valid_init_op])

            LR = get_learning_rate(args.learning_rate, args.learning_rate_step, i)
            for k in range(train_batches_per_epoch):
                # Train
                ops_list = [train_op, train_loss, train_accuracy, global_step, train_summary, train_targets]
                feed_dict = {learning_rate: LR}
                session_output = sess.run(ops_list, feed_dict=feed_dict)

                current_train_loss = session_output[1]
                current_train_accuracy = session_output[2]
                current_step = session_output[3] 
                current_train_summary = session_output[4]
                train_writer.add_summary(current_train_summary, current_step)
                if k % 50 == 0:
                    print("[{:02d} | {:04d} | {:0.1f}] Loss: {:04f} Acc: T {:04f}, V {:04f}   LR {:04f}".format(
                          i, current_step, time.time()-start, current_train_loss, current_train_accuracy, 
                          current_valid_accuracy, LR))

            print("...Validating")
            counter = 0
            valid_accuracy_accumulator = 0.
            for __ in range(valid_batches_per_epoch):
                acc = sess.run(valid_accuracy)
                valid_accuracy_accumulator += acc
                counter += 1

            current_valid_accuracy = valid_accuracy_accumulator / counter
            current_valid_summary = sess.run(valid_summary, feed_dict={valid_accuracy_ph: current_valid_accuracy})
            valid_writer.add_summary(current_valid_summary, current_step)

            print()
            print("[{:02d} | {:04d} | {:0.1f}] V {:04f}".format(i, current_step, time.time()-start, current_valid_accuracy))
            print()

            # Checkpoints
            save_path = os.path.join(args.save_dir, 'model.ckpt')
            saver.save(sess, save_path, global_step=current_step)
            print('Model saved to {:s}'.format(save_path))

    return current_valid_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="minibatch size", type=int, default=64)
    parser.add_argument("--height", help="input image height", type=int, default=32)
    parser.add_argument("--width", help="input image width", type=int, default=32)
    parser.add_argument("--depth", help="input image depth", type=int, default=32)
    parser.add_argument("--n_channels", help="number of input image channels", type=int, default=1)
    parser.add_argument("--n_channels_out", help="number of channel in first layer", type=int, default=32)
    parser.add_argument("--kernel_size", help="number of channel in first layer", type=int, default=3)
    parser.add_argument("--first_kernel_size", help="number of channel in first layer", type=int, default=3)
    parser.add_argument("--n_classes", help="number of output classes", type=int, default=10)

    parser.add_argument("--n_epochs", help="number of minibatches to pass", type=int, default=25)
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=1e-3)
    parser.add_argument("--learning_rate_step", help="interval to divide learning rate by 10", type=int, default=5)

    parser.add_argument("--jitter", help="jitter magnitude", type=int, default=2)

    parser.add_argument("--preprocess", help="whether to preprocess images", type=bool, default=True)
    parser.add_argument("--min_after_dequeue", help="minimum number of images to keep in RAM", type=int, default=500)

    parser.add_argument("--train_file", help="directory of training addresses", default="./addresses/modelnet10_train_addresses.txt")
    parser.add_argument("--valid_file", help="directory of validation addresses", default="./addresses/modelnet10_test_addresses.txt")
    parser.add_argument("--save_dir", help="directory to save results", default="./models/VMar4_0/checkpoints")
    parser.add_argument("--log_dir", help="directory to save results", default="./models/VMar4_0/logs")

    parser.add_argument("--path_increment", help="whether to automatically create new incremented directory", action="store_true")
    parser.add_argument("--save_interval", help="number of iterations between saving model", type=int, default=500)
    parser.add_argument("--delete_existing", help="delete existing models and logs in same folders", action="store_true")
    train(parser.parse_args())


