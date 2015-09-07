#!/usr/bin/python

import cPickle
import numpy as np

fileName='/local/cifar10/'

def unpickle(file):
    fo = open(file,'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

if __name__ == '__main__':
    data = [ unpickle('cifar-10-batches-py/data_batch_' + str(x)) for x in xrange(1,6) ]
    test = unpickle('cifar-10-batches-py/test_batch')

    merged_train_x = np.concatenate([x['data'] for x in data[0:3]], axis = 0)
    merged_train_y = np.concatenate([x['labels'] for x in data[0:3]], axis = 0)
    valid_x = data[4]['data']
    valid_y = data[4]['labels']
    test_x = test['data']
    test_y = test['labels']
    print "... saving"
    np.savez_compressed(fileName + 'train',x=merged_train_x,y=merged_train_y)
    np.savez_compressed(fileName + 'valid',x=valid_x,y=valid_y)
    np.savez_compressed(fileName + 'test',x=test_x,y=test_y)
