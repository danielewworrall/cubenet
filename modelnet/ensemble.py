import os
import sys
import time

import numpy as np
import pickle as pkl


def load(fname):
    with open(fname, 'rb') as fp:
        data = pkl.load(fp)
    return data


def get_label(name):
    label = name.split('/')[3].split('.')[0] 
    return int(label)-1


def dict_add(my_dict, key, val):
    if key not in my_dict:
        my_dict[key] = []
    my_dict[key].append(val)
    return my_dict


def main(fnames):
    # Agregate
    my_dict = {}
    for fname in fnames:
        data = load(fname)
        for key, val in data.items():
            my_dict = dict_add(my_dict, key, val)    

    # Accuracy 
    accuracy_avg = 0.
    accuracy = 0.
    counter = 0.
    for key, val in my_dict.items():
        label = get_label(key)
        val = np.stack(val, 0)
        #val = np.reshape(val, [val.shape[0], -1,10])

        pred = np.argmax(val, 2)
        val_avg = np.mean(val, 0)
        pred_avg = np.argmax(val_avg, 1)

        accuracy += np.mean(np.equal(label, pred), 1)
        accuracy_avg += np.mean(np.equal(label, pred_avg))
        counter += 1

    print("acc: {}".format(accuracy / counter))
    print("avg: {}".format(accuracy_avg / counter))


if __name__ == "__main__":
    fnames = ["./VMar9_0.pkl", './VMar8_1.pkl', './VMar8_2.pkl', './VMar8_3.pkl']
    main(fnames)






























