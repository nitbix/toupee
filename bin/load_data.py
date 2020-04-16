import sys
import os
import argparse
import numpy as np
import tensorflow as tf

def download_cifar10():
    data = tf.keras.datasets.cifar10.load_data()
    return {
        'train': data[0],
        'valid': data[1],
        'test' : data[1]
    }

def download_cifar100():
    data = tf.keras.datasets.cifar10.load_data()

    return {
        'train': data[0],
        'valid': data[1],
        'test' : data[1]
    }

MAPPING = {
    'cifar10': download_cifar10,
    'cifar-10': download_cifar10,
    'cifar100': download_cifar100,
    'cifar-100': download_cifar100
}

def main(args=None):
    """ Train a base model as specified """
    if args is None:
        parser = argparse.ArgumentParser(description='Train a single Base Model')
        parser.add_argument('dataset', help='the dataset name')
        parser.add_argument('save_dir', nargs='?',
                            help='the location where to save the dataset')
        args = parser.parse_args()
    data = MAPPING[args.dataset]()
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    for set_name, set_data in data.items():
        np.savez(os.path.join(args.save_dir, set_name + '.npz'), x=set_data[0], y=set_data[1])

if __name__ == '__main__':
    main()