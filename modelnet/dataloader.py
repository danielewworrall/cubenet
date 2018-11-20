"""Data-loader using new TF Dataset class for Plankton"""
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf


class DataLoader(object):
    """Wrapper class around TF dataset pipeline"""
    
    def __init__(self, address_file, mode, batch_size, height, jitter, 
                 shuffle=True, buffer_size=1000, num_threads=8):
        """Create a new dataloader for the Kaggle plankton dataset

        Args:
            address_file: path to file of addresses and labels
            mode: 'train' or 'test'
            batch_size: int for number of images per batch
            n_classes: int for number of classes in dataset
            height: output image height
            width: output image width
            shuffle: bool for whether to shuffle dataset order
            buffer_size: int for number of images to store in buffer
            num_threads: int for number of CPU threads for preprocessing
        Raises:
            ValueError: If an invalid mode is passed
        """
        self.address_file = address_file
        self.height = height
        self.jitter = jitter

        # Read in data from address file
        self._read_address_file(shuffle)
        self.n_classes = len(np.unique(self.labels))
        self.data_size = len(self.labels)
        print("{} classes detected".format(self.n_classes))
        print("{} training examples detected".format(self.data_size))

        # Tensorize data
        self.img_paths = tf.convert_to_tensor(self.img_paths, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.labels, dtype=tf.string)
        self.labels = tf.string_to_number(self.labels, out_type=tf.int32)-1


        # Create TF dataset object
        data = tf.data.Dataset.from_tensor_slices((self.img_paths, self.labels))
        
        if mode == 'train':
            data = data.map(self._preprocess_train, num_parallel_calls=8).prefetch(100*batch_size)
        elif mode == 'test':
            data = data.map(self._preprocess_test, num_parallel_calls=8).prefetch(10*batch_size)
        else:
            raise ValueError("Invalid mode '{:s}'.".format(mode))

        # Shuffle within buffer for training
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        # Minibatch
        self.data = data.batch(batch_size)

    
    def _read_address_file(self, shuffle):
        """Read contents of address file and store in lists"""
        self.img_paths = []
        self.labels = []
        with open(self.address_file, 'r') as fp:
            lines = fp.readlines()
            if shuffle:
                random.shuffle(lines) 

            for line in lines:
                items = line.replace('\n','').split(',')
                self.img_paths.append(items[0])
                self.labels.append(items[1])


    def _preprocess_train(self, filename, label):
        """"Input preprocessing for training mode"""
        img_string = tf.read_file(filename)
        image = tf.image.decode_image(img_string, channels=1)
        image = 6.*tf.to_float(tf.stack([image]))-1.
        image.set_shape([self.height, self.height, self.height])
        
        # Data augmentation
        # x-flip
        image = tf.image.random_flip_up_down(image)
        # y-flip
        image = tf.image.random_flip_left_right(image)
        # x-y-z jitter
        J = self.jitter
        paddings = tf.constant([[J,J],[J,J],[J,J],[0,0]])
        image = tf.reshape(image, [self.height, self.height, self.height,1])
        image = tf.pad(image, paddings)
        image = tf.random_crop(image, tf.constant([self.height, self.height, self.height, 1]))
        image = tf.reshape(image, [self.height, self.height, self.height, 1])
        
        return image, label


    def _preprocess_test(self, filename, label):
        """"Input preprocessing for training mode"""
        img_string = tf.read_file(filename)
        image = tf.image.decode_image(img_string, channels=1)
        image = 6.*tf.to_float(tf.stack([image]))-1.
        image.set_shape([self.height, self.height, self.height ,1])
        
        return image, label

