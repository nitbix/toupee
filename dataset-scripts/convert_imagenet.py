import h5py
import os
import sys
import numpy as np
import argparse
from PIL import Image


def spinning_cursor():
    while True:
        for cursor in '|/-\\':
            yield cursor


spinner = spinning_cursor()


#tinyImagenet is 64x64, regular Imagenet is variable and needs to be resized
TARGET_SHAPE = (3,64,64)
SIZE = (64,64)


def crops(path, SIZE):
    c = []
    img = Image.open(path)
    sz = img.size
    middle_x = (sz[0] - SIZE[0]) / 2
    middle_y = (sz[1] - SIZE[1]) / 2
    l = max(0, sz[0] - SIZE[0])
    t = max(0, sz[1] - SIZE[1])
    x_pad = max(0, SIZE[0] - sz[0])
    y_pad = max(0, SIZE[1] - sz[1])
    thumb = img.copy()
    thumb.thumbnail(SIZE, Image.ANTIALIAS)
    c.append(thumb.crop((0, 0, SIZE[0], SIZE[1])))
#    c.append(img.crop((middle_x, middle_y,
#        SIZE[0] + middle_x, SIZE[1] + middle_y)))
#    c.append(img.crop((0, 0, SIZE[0], SIZE[1])))
#    c.append(img.crop((0, t, SIZE[0], SIZE[1] + t)))
#    c.append(img.crop((l, 0, SIZE[0] + l, SIZE[1])))
#    c.append(img.crop((l, t, SIZE[0] + l, SIZE[1] + t)))
    return c


#TODO: refactor to pass in h5 object and write directly to that
def make_array(target_dir):
    X = []
    for f in os.listdir(target_dir):
        if 'JPEG' in f:
            for j in crops(os.path.join(target_dir, f), SIZE):
                x = np.transpose(np.array(j.convert('RGB')), (2, 1, 0))
                if x.shape == TARGET_SHAPE:
                    X.append(x)
                else:
                    X.append(np.repeat(x, TARGET_SHAPE[-1]).reshape(TARGET_SHAPE))
            sys.stdout.write('\b')
            sys.stdout.write(next(spinner))
            sys.stdout.flush()
    return np.asarray(X)


def make_labelled(target_dir):
    with h5py.File(target_dir + '.h5', 'w') as hf:
        n_classes = len(os.listdir(os.path.join(target_dir, 'images')))
        for class_num, class_dir in enumerate(
                sorted(os.listdir(os.path.join(target_dir, 'images')))):
            X_c = make_array(os.path.join(target_dir, 'images', class_dir))
            y_c = np.array([class_num] * X_c.shape[0])
            if 'X' not in hf:
                sh = list(X_c.shape)
                sh[0] = None
                hf.create_dataset('X', data=X_c, maxshape=sh)
                hf.create_dataset('y', data=y_c, maxshape=(None, ))
            else:
                hf['X'].resize((hf['X'].shape[0] + X_c.shape[0]), axis=0)
                hf['X'][-X_c.shape[0]:] = X_c
                hf['y'].resize((hf['y'].shape[0] + y_c.shape[0]), axis=0)
                hf['y'][-y_c.shape[0]:] = y_c
        assert hf['X'].shape[0] == hf['y'].shape[0]
        print(hf['X'].shape)


def make_unlabelled(target_dir):
    with h5py.File(target_dir + '.h5', 'w') as hf:
        X = make_array(target_dir)
        hf.create_dataset('X', data=X)
        print(hf['X'].shape)


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
