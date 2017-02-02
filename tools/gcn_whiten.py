#!/usr/bin/python

from toupee import data
import numpy as np
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert a pylearn2 dataset')
    parser.add_argument('--dest', help='the destination for the dataset')
    parser.add_argument('--source', help='the source of the data')
    parser.add_argument('--shape', help='new shape to enforce')
    parser.add_argument('--tf-to-th', help='reorder dimensions from tf to th',
            action='store_true')
    parser.add_argument('--th-to-tf', help='reorder dimensions from th to tf',
            action='store_true')
    args = parser.parse_args()

    dataset = data.load_data(args.source,
                             pickled = False,
                             one_hot_y = False,
                             center_and_normalise = True,
                             join_train_and_valid = False,
                             zca_whitening = True)


    np.savez_compressed(args.dest + "/train", x=dataset[0][0], y=dataset[0][1])
    np.savez_compressed(args.dest + "/valid", x=dataset[1][0], y=dataset[1][1])
    np.savez_compressed(args.dest + "/test", x=dataset[2][0], y=dataset[2][1])
