#!/usr/bin/python

import cPickle
import numpy as np
import argparse
from toupee.data import one_hot
from matplotlib import pyplot

def unpickle(file):
    fo = open(file,'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert a pylearn2 dataset')
    parser.add_argument('--dest', help='the destination for the dataset')
    parser.add_argument('--source', help='the source of the data')
    args = parser.parse_args()

    print "... saving"
    x = unpickle(args.source)
    img = x['data'][20].reshape((3,32,32))
    img = np.transpose(img, (1,2,0))
    pyplot.imshow(img)
    pyplot.show()
    exit(1)
    x = np.load(args.source + ".npy")
    if args.shape is not None:
        shape = np.asarray(map(int, args.shape.split(",")))
        print "reshaping to {0}".format(shape)
        x = x.reshape(shape)
    if args.tf_to_th:
        print "tf to th"
        x = np.transpose(x, (0, 3, 1, 2))
    if args.th_to_tf:
        print "th to tf"
        x = np.transpose(x, (0, 2, 3, 1))
    y = unpickle(args.source + ".pkl")
    y = one_hot(np.asarray(y.y))
    print x[0].shape
    pyplot.imshow(x[0])
    pyplot.show()
    exit(1)
    np.savez_compressed(args.dest, x=x, y=y)
