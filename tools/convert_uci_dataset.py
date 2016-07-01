#!/usr/bin/python

import csv
import argparse
import numpy

def split_label_and_data(data,label_first):
    if label_first:
        return data[0],data[1:]
    else:
        return data[:len(data) - 1],data[len(data) - 1]

def convert_input(row,column_index = {}):
    ret = []
    i = 0
    for entry in row:
        try:
            ret.append(float(entry))
        except:
            if i not in column_index.keys():
                column_index[i] = {}
            if entry not in column_index[i].keys():
                if len(column_index[i].values()) > 0:
                    next_value = max(column_index[i].values()) + 1
                    #print "{0} = {1}".format(entry,next_value)
                else:
                    next_value = 0.
                column_index[i][entry] = next_value
            
            ret.append(float(column_index[i][entry]))
        i += 1
    return ret

def convert_label(entry,column_index = {}):
    i = 0
    try:
        return float(entry)
    except:
        if entry not in column_index.keys():
            if len(column_index.values()) > 0:
                next_value = max(column_index.values()) + 1
            else:
                next_value = 0.
            column_index[entry] = next_value
        return float(column_index[entry])

def csv_to_numpy(file_name,labels_first,offset=0.):
    with open(file_name, 'rb') as csvfile:
        X = []
        Y = []
        X_column_index = {}
        Y_column_index = {}
        reader = csv.reader(csvfile, delimiter=',')
        start_line = 0
        if args.header:
            start_line = 1
        line_count = 0
        for row in reader:
            if line_count > start_line and len(row) > 0:
                x,y = split_label_and_data(row,labels_first)
                X.append(convert_input(x,X_column_index))
                Y.append(convert_label(y,Y_column_index))
            line_count += 1
    if args.labels_offset != 0:
        Y = numpy.add(Y,args.labels_offset)
    shuffled_X = []
    shuffled_Y = []
    for i in numpy.random.permutation(len(X)):
        shuffled_X.append(numpy.asarray(X[i]))
        shuffled_Y.append(numpy.asarray(Y[i]))

    return numpy.asarray(shuffled_X),numpy.asarray(shuffled_Y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a UCI dataset')
    parser.add_argument('--dest', help='the destination for the dataset')
    parser.add_argument('--source', help='the source of the data')
    parser.add_argument('--source-test', nargs='?',
            help='optional test set source')
    parser.add_argument('--source-valid', nargs='?',
            help='optional valid set source (only works if you have test too)')
    parser.add_argument('--header', action='store_true', help='has header line')
    parser.add_argument('--labels-first', action='store_true',
            help='labels are in the first column instead of the last')
    parser.add_argument('--labels-offset', type=int, default=0,
            help='fixed int to remove from the label')
    args=parser.parse_args()

    X,Y = csv_to_numpy(args.source,args.labels_first,args.labels_offset)
    data_size = len(X)
    if args.source_test is not None:
        train_x,train_y = X,Y
        if args.source_valid is not None:
            valid_x,valid_y = csv_to_numpy(args.source_valid,args.labels_first,args.labels_offset)
        else:
            valid_split = data_size * 0.9
            valid_x = train_x[valid_split:]
            valid_y = train_y[valid_split:]
            train_x = train_x[:valid_split]
            train_y = train_y[:valid_split]
            test_x,test_y = csv_to_numpy(args.source_test,args.labels_first,args.labels_offset)
    else:
        train_split = data_size * 0.8
        test_split = data_size * 0.9
        train_x = X[:train_split]
        train_y = Y[:train_split]
        valid_x = X[train_split:test_split]
        valid_y = Y[train_split:test_split]
        test_x = X[test_split:]
        test_y = Y[test_split:]
    numpy.savez_compressed(args.dest + 'train',x=train_x,y=train_y)
    numpy.savez_compressed(args.dest + 'valid',x=valid_x,y=valid_y)
    numpy.savez_compressed(args.dest + 'test',x=test_x,y=test_y)

    print "{0} features, {1} classes, {2} training, {3} validation, {4} test".format(
        len(train_x[0]),
        int(max(train_y)) + 1,
        len(train_x),
        len(valid_x),
        len(test_x),
    )
