#!/usr/bin/python

import cPickle
import numpy as np
import leveldb
import sys
import argparse
import caffe
import numpy
from caffe.proto import caffe_pb2

def load_file(filename):
    db = leveldb.LevelDB(filename)
    datum = caffe_pb2.Datum()
    X = []
    Y = []
    for key, value in db.RangeIter():
        datum.ParseFromString(value)
        label = datum.label
        data = caffe.io.datum_to_array(datum)
        X.append(data)
        Y.append(label)
    return X,Y

def save(where,set_x,set_y):
    set_x = numpy.asarray(set_x, dtype='float32')
    set_y = numpy.asarray(set_y, dtype='int32')
    np.savez_compressed(where,x=set_x,y=set_y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a Caffe/LMDB dataset')
    parser.add_argument('--dest', help='the destination for the dataset')
    parser.add_argument('--source-train', help='the source of the data')
    parser.add_argument('--source-test', nargs='?',
            help='optional test set source')
    parser.add_argument('--source-valid', nargs='?',
            help='optional valid set source (only works if you have test too)')
    args=parser.parse_args()

    X,Y = load_file(args.source_train)

    print "... saving"
    if args.source_valid is None:
        data_size = len(X)
        shuffled_X = []
        shuffled_Y = []
        for i in numpy.random.permutation(data_size):
            shuffled_X.append(numpy.asarray(X[i]))
            shuffled_Y.append(numpy.asarray(Y[i]))
        train_split = int(data_size * 0.8)
        train_x = numpy.asarray(shuffled_X[:train_split])
        train_y = numpy.asarray(shuffled_Y[:train_split])
        valid_x = numpy.asarray(shuffled_X[train_split:])
        valid_y = numpy.asarray(shuffled_Y[train_split:])
        save(args.dest + '/train',train_x,train_y)
        save(args.dest + '/valid',valid_x,valid_y)
    else:
        train_x,train_y = X,Y
        valid_x,valid_y = load_file(args.source_valid)
        save(args.dest + '/train',train_x,train_y)
        save(args.dest + '/valid',valid_x,valid_y)

    test_x,test_y = load_file(args.source_test)
    save(args.dest + '/test',test_x,test_y)
