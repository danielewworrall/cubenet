import argparse
import os
import random
import sys
import time

import numpy as np
import zlib

from io import BytesIO

PREFIX = 'data/'
SUFFIX = '.npy.z'


# Dataloader
class ModelNetLoader:
    def __init__(self, foldername):
        self.foldername = foldername
        self.canonical_names, self.all_names = self.load_names()


    def load_names(self):
        # Get names of all files
        paths = [os.path.join(self.foldername, f) for f in os.listdir(self.foldername)]
        filenames = [f for f in paths if os.path.isfile(f)]
        print("{} files found".format(len(filenames)))
        
        # Sort names
        canonical_names = []
        all_names = []
        for name in filenames:
            num = int(name.split('.npy')[0].split('.')[-1])
            if num == 1:
                canonical_names.append(name)
            all_names.append(name)
        return canonical_names, all_names


    def iterator(self, batch_size, augment_data=False, group_rotations=False, shuffle=False):
        """If group_rotations is True, return data in chunks of 12 rotated
        copies of the same item
        Args:
            augment_data: bool
            group_rotations: grouping flag
            randomize_order: randomize the data input ordering
        Returns:
            a [n_rotations, height, width, depth, 1] array
        """
        if group_rotations:
            names = self.canonical_names.copy()
        else:
            names = self.all_names.copy()

        # Per-epoch data shuffling
        if shuffle:
            random.shuffle(names)

        images_list = []
        labels_list = []
        for name in names:
            # Get label name
            label = name.split(self.foldername)[1].replace('/','').split('.')[0]
            label = int(label)-1

            if group_rotations:
                to_load = []
                for i in range(12):
                    split_name = name.split('.npy')
                    resplit_name = split_name[0].split('.')[:-1]
                    new_rotation = "{}.{:03d}{}{}".format('.'.join(resplit_name), i+1, '.npy', split_name[-1])
                    to_load.append(new_rotation)
            else:
                to_load = [name,]
                
            image_batch = []
            label_batch = []
            for filename in to_load:
                with open(name, 'rb') as fp:
                    buf = zlib.decompress(fp.read())
                    x = np.load(BytesIO(buf))
    
                # As per Brock et al, 2016
                x = 6.*x - 1.
                # Data augmentation
                if augment_data == True:
                    x = self.augment(x)
                image_batch.append(x)
                label_batch.append(label)
            images_list.extend(image_batch)
            labels_list.extend(label_batch)

            if len(images_list) >= batch_size:
                images = np.stack(images_list, 0)
                labels = np.stack(labels_list, 0).astype(np.int32)
                images_list = []
                labels_list = []
                yield images[...,np.newaxis], labels

        # Yield a smaller minibatch
        if images_list:
            images = np.stack(images_list, 0)
            labels = np.stack(labels_list, 0).astype(np.int32)
            images_list = []
            labels_list = []
            yield images[...,np.newaxis], labels


    def augment(self, x):
        """Data augmentation as per Brock et al. 2017"""
        # Random x-y flips
        if np.random.binomial(1, .2):
            x = np.flip(x, 0)
        if np.random.binomial(1, .2):
            x = np.flip(x, 1)
        for i in range(3):
            x = np.roll(x, np.random.random_integers(-2,2), i)
        return x


if __name__ == '__main__':
    dataset_obj = ModelNet10Loader("./data/shapenet10_train")
    dataset_obj.data_iterator(shuffle=True, group_rotations=True)

