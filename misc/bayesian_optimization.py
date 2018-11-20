import argparse
import os
import sys
import time

import numpy as np

from train import train
from skopt import gp_minimize


def optimize(n_calls):
    """Run gp_minimize"""
    dimensions = [(16,64),                      # batch size
                  (10,25),                      # n_epochs
                  [16,32,48,64],                # n_channels_out
                  (1e-4, 1e-2, 'log-uniform'),  # learning rate
                  (2,15),                       # learning rate step
                  (1,5)]                        # jitter
    x0 = [32,15,32,1e-3,5,2]

    print(gp_minimize(wrapper_function, dimensions, x0=x0, n_calls=n_calls, verbose=True))


def wrapper_function(dimensions):
    """CNN wrapper"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--height", help="input image height", type=int, default=32)
    parser.add_argument("--width", help="input image width", type=int, default=32)
    parser.add_argument("--depth", help="input image depth", type=int, default=32)
    parser.add_argument("--n_channels", help="number of input image channels", type=int, default=1)
    parser.add_argument("--kernel_size", help="number of channel in first layer", type=int, default=3)
    parser.add_argument("--first_kernel_size", help="number of channel in first layer", type=int, default=3)
    parser.add_argument("--n_classes", help="number of output classes", type=int, default=40)

    parser.add_argument("--preprocess", help="whether to preprocess images", type=bool, default=True)
    parser.add_argument("--min_after_dequeue", help="minimum number of images to keep in RAM", type=int, default=500)

    parser.add_argument("--train_file", help="directory of training addresses", default="./addresses/modelnet40_train_addresses.txt")
    parser.add_argument("--valid_file", help="directory of validation addresses", default="./addresses/modelnet40_test_addresses.txt")
    parser.add_argument("--save_dir", help="directory to save results", default="./models/VMar4B_0/checkpoints")
    parser.add_argument("--log_dir", help="directory to save results", default="./models/VMar4B_0/logs")

    parser.add_argument("--save_interval", help="number of iterations between saving model", type=int, default=500)

    args = parser.parse_args()

    args.batch_size = int(dimensions[0])
    args.n_epochs = int(dimensions[1])
    args.n_channels_out = int(dimensions[2])
    args.learning_rate = float(dimensions[3])
    args.learning_rate_step = int(dimensions[4])
    args.jitter = int(dimensions[5])

    args.path_increment = True
    args.delete_existing = False

    for arg in vars(args):
        print(arg, getattr(args, arg))
    print()

    valid_acc = train(args)
    error = 1. - valid_acc
    print("Error: {}".format(error))

    return error


if __name__ == "__main__":
    optimize(12)
