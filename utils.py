"""Common utility functions"""

import os
import shutil

import numpy as np
import tensorflow as tf
import json
import logging


def make_dirs(args, directory):
    """Make new directory

    Make a new directory to store results of the current experiment. If it exists
    already query the user to overwrite, unless args.delet_existing is set to True,
    in which case, delete without asking. If the folder name is of the form
    my_dir/my_subdir_<num>, and args.path_increment is set to True, then create 
    new folder with name my_dir/my_subdir_<num+1>

    Args:
       args: argument namespace
       directory: address of directory

    Returns:
       name of directory created
    """
    if directory is not None:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info('Created {:s}'.format(directory))
        else:
            if args.path_increment:
                while(os.path.exists(directory)):
                    split = directory.split('_')
                    resplit = split[1].split('/')
                    directory = split[0] + '_' + str(int(resplit[0]) + 1)
                    directory = os.path.join(directory, resplit[1])
                os.makedirs(directory)
                logging.info('Created {:s}'.format(directory))
            else:
                if not args.delete_existing:
                    input('{:s} already exists: Press ENTER to overwrite contents.'
                          .format(directory))
                shutil.rmtree(directory)
                os.makedirs(directory)
    return directory


def manage_directories(args):
    """Directory manager

    Make all directories needed for saving results/models.

    Args:
        args: argparse namespace
    Returns:
        new argparse namespace with updated save and log directories
    """
    args.save_dir = make_dirs(args, args.save_dir)
    args.log_dir = make_dirs(args, args.log_dir)
    return args
