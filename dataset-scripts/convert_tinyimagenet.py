import h5py
import os
import numpy as np
import argparse
from PIL import Image

#tinyImagenet is 64x64, regular Imagenet is variable and needs to be resized
TARGET_SHAPE = (64,64,3)

#TODO: refactor to pass in h5 object and write directly to that
def make_array(target_dir):
    X = []
    for f in os.listdir(os.path.join(target_dir, 'images')):
        if 'JPEG' in f:
            x = np.asarray(Image.open(os.path.join(target_dir, 'images', f)))
            if x.shape == TARGET_SHAPE:
                X.append(x)
            else:
                X.append(np.repeat(x, TARGET_SHAPE[-1]).reshape(TARGET_SHAPE))
    return np.asarray(X)


def make_labelled(target_dir):
    X = [] #TODO: create h5 here
    y = []
    for class_dir in os.listdir(target_dir):
        X_c = make_array(os.path.join(target_dir, class_dir))
        y_c = [class_dir] * X_c.shape[0]
        X.extend(X_c) #TODO: append to h5
        y.extend(y_c)
    X = np.asarray(X)
    y = np.asarray(y)
    assert X.shape[0] == y.shape[0]
    np.savez(target_dir + '.npz', X = X, y = y)


def make_unlabelled(target_dir):
    #TODO: save to h5
    X = make_array(target_dir)
    np.savez(target_dir + '.npz', X = X)


if __name__ == "__main__":
    #TODO: --format=h5
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-folder', help="Taget folder, if not in live mode.", 
            default=os.path.expanduser("/local/data/tinyImagenet/"))
    args = parser.parse_args()
    os.chdir(args.target_folder)
    make_labelled('train')
    make_unlabelled('val')
    make_unlabelled('test')
