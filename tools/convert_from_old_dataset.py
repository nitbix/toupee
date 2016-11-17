#!/usr/bin/python
import sys
import numpy as np
import os
from toupee import data

if __name__ == '__main__':
    location = sys.argv[1]
    dest = sys.argv[1] + "_th/"
    if not os.path.exists(dest):
        os.mkdir(dest)
    shape = [int(x) for x in sys.argv[2].split(',')]
    dataset = data.load_data(location,
                             pickled = False,
                             one_hot_y = True)
    print dataset[0][0].shape
    np.savez_compressed(dest + 'train',
            x=dataset[0][0].reshape([dataset[0][0].shape[0]] + shape),
            y=dataset[0][1])
    np.savez_compressed(dest + 'valid',
            x=dataset[1][0].reshape([dataset[1][0].shape[0]] + shape),
            y=dataset[1][1])
    np.savez_compressed(dest + 'test',
            x=dataset[2][0].reshape([dataset[2][0].shape[0]] + shape),
            y=dataset[2][1])
