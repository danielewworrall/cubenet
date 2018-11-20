import argparse
import os
import random
import sys
import time

import numpy as np
import zlib

from io import BytesIO
from skimage.io import imsave

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


    def iterator(self):
        """If group_rotations is True, return data in chunks of 12 rotated
        copies of the same item

        Args:
            augment_data: bool
            group_rotations: grouping flag
            randomize_order: randomize the data input ordering
        Returns:
            a [n_rotations, height, width, depth, 1] array
        """
        names = self.all_names.copy()

        images_list = []
        labels_list = []
        for name in names:
            with open(name, 'rb') as fp:
                buf = zlib.decompress(fp.read())
                x = np.load(BytesIO(buf))
                x = np.reshape(x, [32,32*32])
                new_name = name.split('.z')[0]
                new_name = new_name.replace("data","decompressed")
                new_name = new_name.replace(".npy", ".png")
                imsave(new_name, x)
    


if __name__ == '__main__':
    dataset_obj = ModelNetLoader("./data/modelnet40_test")
    dataset_obj.iterator()
