import os
import sys
import time


def get_files(folder):
    files = os.listdir(folder)
    files = [os.path.join(folder, f) for f in files]
    files = [f for f in files if os.path.isfile(f)]
    return files


def write_files(folder, filename):
    files = get_files(folder)
    labels = [f.split('/')[-1].split('.')[0] for f in files]
    lines = ["{},{}\n".format(f,label) for (f, label) in zip(files, labels)]
    print("Writing")
    with open(filename, 'w') as fp:
        fp.writelines(lines)


if __name__ == "__main__":
    filename = "./addresses/modelnet40_test_addresses.txt"
    folder = "./decompressed/modelnet40_test"
    write_files(folder, filename)
