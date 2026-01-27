import sys
import os
import argparse
import numpy as np # type: ignore
from torchvision import datasets
from toupee.utils import dict_map


def _rgb_to_float(features: np.ndarray):
    """ Convert RGB values to float """
    return features / np.max(features)


def _preprocess_cifar(data: dict) -> dict:
    return dict_map(data, lambda d: (_rgb_to_float(d[0]), d[1]))


def _preprocess_mnist(data:dict) -> dict:
    data = dict_map(data, lambda d: (np.expand_dims(d[0], axis=3), d[1]))
    return dict_map(data, lambda d: (_rgb_to_float(d[0]), d[1]))


def download_cifar10() -> dict:
    train_data = datasets.CIFAR10(root='/tmp/data', train=True, download=True)
    test_data = datasets.CIFAR10(root='/tmp/data', train=False, download=True)

    # Convert to numpy arrays
    train_x = np.array(train_data.data)
    train_y = np.array(train_data.targets)
    test_x = np.array(test_data.data)
    test_y = np.array(test_data.targets)

    return _preprocess_cifar({
        'train': (train_x, train_y),
        'valid': (test_x, test_y),
        'test' : (test_x, test_y)
    })


def download_cifar100() -> dict:
    train_data = datasets.CIFAR100(root='/tmp/data', train=True, download=True)
    test_data = datasets.CIFAR100(root='/tmp/data', train=False, download=True)

    # Convert to numpy arrays
    train_x = np.array(train_data.data)
    train_y = np.array(train_data.targets)
    test_x = np.array(test_data.data)
    test_y = np.array(test_data.targets)

    return _preprocess_cifar({
        'train': (train_x, train_y),
        'valid': (test_x, test_y),
        'test' : (test_x, test_y)
    })


def download_mnist() -> dict:
    train_data = datasets.MNIST(root='/tmp/data', train=True, download=True)
    test_data = datasets.MNIST(root='/tmp/data', train=False, download=True)

    # Convert to numpy arrays
    train_x = np.array(train_data.data)
    train_y = np.array(train_data.targets)
    test_x = np.array(test_data.data)
    test_y = np.array(test_data.targets)

    return _preprocess_mnist({
        'train': (train_x, train_y),
        'valid': (test_x, test_y),
        'test' : (test_x, test_y)
    })


MAPPING = {
    'cifar10': download_cifar10,
    'cifar-10': download_cifar10,
    'cifar100': download_cifar100,
    'cifar-100': download_cifar100,
    'mnist': download_mnist,
}


def main(args=None) -> None:
    """ Download a well-known dataset """
    parser = argparse.ArgumentParser(description='Download a well-know dataset')
    parser.add_argument('dataset', help='the dataset name')
    parser.add_argument('save_dir', nargs='?',
                        help='the location where to save the dataset')
    args = parser.parse_args(args)
    print(args.dataset)
    data = MAPPING[args.dataset]()
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    for set_name, set_data in data.items():
        np.savez(os.path.join(args.save_dir, set_name + '.npz'), x=set_data[0], y=set_data[1])


if __name__ == '__main__':
    main()
