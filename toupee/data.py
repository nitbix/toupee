#!/usr/bin/python


"""
Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'

#TODO: Polymorphism for different data formats
#TODO: Online image transformations

import os
import sys
import math
import copy
from typing import Optional, Tuple, Dict, Any, Callable

from pathlib import Path
import numpy as np # type: ignore

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torchvision import transforms

from toupee.utils import dict_map

KNOWN_DATA_TYPES = [".tfrecord", ".h5", ".npz"]

DEFAULT_TRAINING_FILE = 'train'
DEFAULT_VALIDATION_FILE = 'valid'
DEFAULT_TESTING_FILE = 'test'


def get_data_format(filename: str) -> str:
    """ Identifies the extension of the dataset """
    extension = Path(filename).suffix
    if extension not in KNOWN_DATA_TYPES:
        raise ValueError("Unknown data type %s" % extension)
    return extension


def one_hot_numpy(dataset):
    """ Convert a numpy array of Y labels into one-hot encodings """
    n_classes = dataset.max() + 1
    return np.eye(n_classes)[dataset.reshape(-1)]


def _load_h5(filename, **kwargs):
    """ Load an HDF5 file """
    #TODO: write me
    raise NotImplementedError()


def _load_npz(filename: str, **kwargs) -> tuple:
    """ Load a NPZ file """
    #TODO: transformations
    #TODO: special dict mappings
    data = np.load(filename)
    data = (data['x'],data['y'])
    if kwargs['convert_labels_to_one_hot']:
        data = (data[0], one_hot_numpy(data[1]))
    return data


def _load_tfrecord(filename, **kwargs):
    """ Load a TFRecord file """
    #TODO: write me
    raise NotImplementedError()


def load(filename: str, **kwargs) -> tuple:
    """ Load any known data format """
    mapper = {'.h5': _load_h5,
              '.npz': _load_npz,
              '.tfrecord': _load_tfrecord
             }
    return mapper[get_data_format(filename)](filename, **kwargs)


class ToupeeDataset(TorchDataset):
    """
    PyTorch Dataset wrapper for toupee data.

    Handles numpy arrays and optional transforms. Converts from NHWC (Keras default)
    to NCHW (PyTorch default) format for image data.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, transform: Optional[Callable] = None,
                 to_channels_first: bool = True):
        """
        Args:
            x: Input data as numpy array
            y: Labels as numpy array
            transform: Optional torchvision transform to apply
            to_channels_first: If True, permute image data from NHWC to NCHW
        """
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        self.transform = transform
        self.to_channels_first = to_channels_first
        # Check if data is image-like (4D: N, H, W, C)
        self.is_image = len(self.x.shape) == 4

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.x[idx]
        y = self.y[idx]

        # Convert to tensor
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        # Convert NHWC to NCHW for image data
        if self.is_image and self.to_channels_first:
            x = x.permute(2, 0, 1)  # HWC -> CHW

        # Apply transforms
        if self.transform:
            x = self.transform(x)

        return x, y


class ImageAugmentation:
    """
    Image augmentation class that mimics Keras ImageDataGenerator functionality.

    Uses torchvision transforms internally.
    """

    def __init__(self, params: Dict[str, Any], training_data: np.ndarray):
        """
        Args:
            params: Dictionary of augmentation parameters matching Keras ImageDataGenerator
            training_data: Training data used to compute mean/std for normalization
        """
        self.params = params
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

        # Compute featurewise statistics if needed
        if params.get('featurewise_center', False) or params.get('featurewise_std_normalization', False):
            self._fit(training_data)

        # Build transform pipeline
        self.train_transform = self._build_train_transform()
        self.eval_transform = self._build_eval_transform()

    def _fit(self, data: np.ndarray):
        """Compute mean and std from training data."""
        # Data is in NHWC format
        # Compute per-channel mean and std
        if len(data.shape) == 4:
            # Image data: compute over N, H, W dimensions
            self.mean = data.mean(axis=(0, 1, 2))
            self.std = data.std(axis=(0, 1, 2))
        else:
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0)

        # Avoid division by zero
        self.std = np.where(self.std == 0, 1.0, self.std)

    def _build_train_transform(self) -> transforms.Compose:
        """Build training augmentation transform."""
        transform_list = []

        # Random affine transforms
        translate = None
        if 'width_shift_range' in self.params or 'height_shift_range' in self.params:
            width_shift = self.params.get('width_shift_range', 0)
            height_shift = self.params.get('height_shift_range', 0)
            translate = (width_shift, height_shift)

        rotation = self.params.get('rotation_range', 0)

        if translate or rotation:
            transform_list.append(
                transforms.RandomAffine(
                    degrees=rotation,
                    translate=translate
                )
            )

        # Random horizontal flip
        if self.params.get('horizontal_flip', False):
            transform_list.append(transforms.RandomHorizontalFlip())

        # Random vertical flip
        if self.params.get('vertical_flip', False):
            transform_list.append(transforms.RandomVerticalFlip())

        # Zoom (using RandomResizedCrop as approximation)
        zoom_range = self.params.get('zoom_range', 0)
        if zoom_range:
            if isinstance(zoom_range, (int, float)):
                zoom_range = (1 - zoom_range, 1 + zoom_range)
            # Note: this is an approximation, proper zoom would need custom implementation

        # Normalization
        if self.mean is not None and self.std is not None:
            # Convert to torch tensors for normalization
            # Note: mean/std are per-channel, applied after CHW conversion
            mean_tensor = torch.from_numpy(self.mean).float()
            std_tensor = torch.from_numpy(self.std).float()
            transform_list.append(NormalizeTensor(mean_tensor, std_tensor))

        return transforms.Compose(transform_list) if transform_list else None

    def _build_eval_transform(self) -> transforms.Compose:
        """Build evaluation transform (normalization only)."""
        transform_list = []

        if self.mean is not None and self.std is not None:
            mean_tensor = torch.from_numpy(self.mean).float()
            std_tensor = torch.from_numpy(self.std).float()
            transform_list.append(NormalizeTensor(mean_tensor, std_tensor))

        return transforms.Compose(transform_list) if transform_list else None

    def standardize(self, data: np.ndarray) -> np.ndarray:
        """Standardize data using computed mean/std. Used for inference."""
        if self.mean is not None:
            data = data - self.mean
        if self.std is not None:
            data = data / self.std
        return data


class NormalizeTensor:
    """Normalize a tensor with mean and standard deviation."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor is CHW, mean/std are per-channel
        # Reshape mean/std for broadcasting
        if tensor.dim() == 3:  # CHW
            mean = self.mean.view(-1, 1, 1)
            std = self.std.view(-1, 1, 1)
        else:
            mean = self.mean
            std = self.std
        return (tensor - mean) / std


def _resample_np(data: tuple, sample_size: int, weights, replace: bool) -> tuple:
    """ Rasampling specifically for np arrays """
    data_size = data[0].shape[0]
    indices = np.random.choice(range(data_size), size=sample_size, replace=replace, p=weights)
    return tuple(np.take(feature, indices, axis=0) for feature in data)


def _create_dataloader(data: tuple, batch_size: int, shuffle: bool = False,
                       transform: Optional[Callable] = None, **kwargs) -> DataLoader:
    """Create a PyTorch DataLoader from numpy data."""
    dataset = ToupeeDataset(data[0], data[1], transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=kwargs.get('num_workers', 0),
        pin_memory=kwargs.get('pin_memory', True),
        drop_last=False
    )


class Dataset:
    """ Class to load a dataset """
    def __init__(self,
                 src_dir: str = None,
                 training_file: str = None,
                 validation_file: str = None,
                 testing_file: str = None,
                 data_format: str = "",
                 shuffle: bool = False,
                 img_gen_params: dict = None,
                 max_examples: int = None,
                 **kwargs):
        self.files = {}
        self.img_aug: Optional[ImageAugmentation] = None
        self.shuffle = shuffle
        if src_dir is None and training_file is None:
            raise ValueError("Must specify one of src_dir or training_file")
        self.files['train'] = training_file or DEFAULT_TRAINING_FILE
        self.files['valid'] = validation_file or DEFAULT_VALIDATION_FILE
        self.files['test'] = testing_file or DEFAULT_TESTING_FILE
        if src_dir:
            src_dir = os.path.expanduser(src_dir)
            self.files = dict_map(self.files, lambda f_name: os.path.join(src_dir, f_name))
        if data_format:
            self.files = dict_map(self.files, lambda f_name: f_name + '.' + data_format)
        if not os.path.exists(self.files['train']):
            raise ValueError('Training file %s not found!' % self.files['train'])
        self.files = dict_map(self.files, lambda f_name: f_name if os.path.exists(f_name) else None)
        self.data_format = get_data_format(self.files['train'])
        self.img_gen_params = img_gen_params
        for f_name in self.files.values():
            if f_name is not None and get_data_format(f_name) != self.data_format:
                raise ValueError("All files must be in same format")
        self.raw_data = dict_map(self.files,
                                 lambda f_name: load(filename=f_name, **kwargs) if f_name else None)
        self.n_classes = self.raw_data['train'][1].shape[1]
        if max_examples:
            self.raw_data = dict_map(self.raw_data, lambda d: (d[0][:max_examples], d[1][:max_examples]))
        self._process(**kwargs)

    def _process(self, **kwargs):
        self.size = dict_map(self.raw_data, lambda data: data[0].shape[0])
        self.batch_size = kwargs.get('batch_size', 32)
        # Calculate steps_per_epoch for training
        self.steps_per_epoch = dict_map(self.size,
                                        lambda size: math.ceil(float(size) / self.batch_size))

        # Set up image augmentation if specified
        if self.img_gen_params:
            self.img_aug = ImageAugmentation(self.img_gen_params, self.raw_data['train'][0])
            train_transform = self.img_aug.train_transform
            eval_transform = self.img_aug.eval_transform
        else:
            train_transform = None
            eval_transform = None

        # Filter kwargs to remove keys we pass explicitly
        dataloader_kwargs = {k: v for k, v in kwargs.items()
                            if k not in ('batch_size', 'shuffle')}

        # Create DataLoaders
        self.data = {}

        # Training DataLoader with augmentation
        self.data['train'] = _create_dataloader(
            self.raw_data['train'],
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            transform=train_transform,
            **dataloader_kwargs
        )

        # Validation DataLoader with normalization only
        if self.raw_data.get('valid') is not None:
            self.data['valid'] = _create_dataloader(
                self.raw_data['valid'],
                batch_size=self.batch_size,
                shuffle=False,
                transform=eval_transform,
                **dataloader_kwargs
            )

        # Test DataLoader with normalization only
        if self.raw_data.get('test') is not None:
            self.data['test'] = _create_dataloader(
                self.raw_data['test'],
                batch_size=self.batch_size,
                shuffle=False,
                transform=eval_transform,
                **dataloader_kwargs
            )

        # Also create standardized versions for compatibility
        if self.img_aug:
            self.standardized_raw_data = dict_map(
                self.raw_data,
                lambda data: (self.img_aug.standardize(np.copy(data[0])), data[1]) if data else None
            )
            self.standardized_data = {}
            for split in ('valid', 'test'):
                if self.standardized_raw_data.get(split) is not None:
                    self.standardized_data[split] = _create_dataloader(
                        self.standardized_raw_data[split],
                        batch_size=self.batch_size,
                        shuffle=False,
                        **dataloader_kwargs
                    )

        self.kwargs = kwargs

    def get_training_handle(self) -> DataLoader:
        """ Return the DataLoader for training """
        return self.data['train']

    def get_handle(self, split: str, standardized: bool = False) -> DataLoader:
        """ Return the DataLoader for a given split """
        if standardized and self.img_aug and hasattr(self, 'standardized_data'):
            return self.standardized_data.get(split, self.data[split])
        return self.data[split]

    def get_validation_handle(self, standardized: bool = False) -> DataLoader:
        """ Return the DataLoader for validation """
        return self.get_handle(split='valid', standardized=standardized)

    def get_testing_handle(self, standardized: bool = False) -> DataLoader:
        """ Return the DataLoader for testing """
        return self.get_handle(split='test', standardized=standardized)

    def resample(self, sample_size: int = None, weights=None, replace: bool = True):
        """ Promote the dataset to a resampling one """
        self.resample_size = sample_size or self.raw_data['train'][0].shape[0]
        self.resample_weights = weights
        self.resample_replace = replace
        self.__class__ = ResamplingDataset
        return self

    def copy_with_new_data(self, new_data):
        """ Create a new copy but with new data. Useful for distillation and post-processing. """
        new_dataset = copy.copy(self)
        new_dataset.raw_data = new_data
        new_dataset._process(**self.kwargs)
        return new_dataset

    def map(self, f, columns=None):
        """ Map a function that takes a numpy array over the entire dataset. Optionally select which columns. """
        all_cols = range(len(self.raw_data['train']))
        if not columns:
            columns = all_cols
        new_data = dict_map(self.raw_data, lambda x: tuple([f(x[col]) if col in columns else x[col] for col in all_cols]))
        return self.copy_with_new_data(new_data)

    def map_splits(self, f):
        """ Map a function that operates on a train/valid/test split to all three. """
        return self.copy_with_new_data(dict_map(self.raw_data, f))

    def distillation_dataset(self, model):
        return self.map_splits(lambda raw_data: (raw_data[0], model.predict_proba(raw_data[0])))


class ResamplingDataset(Dataset):
    """ A dataset that resamples every time a training handle is requested """

    def __init__(self, **kwargs):
        raise RuntimeError("ResamplingDatasetWrapper cannot be created directly, use Dataset.resample")

    def set_weights(self, weights):
        """ Change the resampling weights """
        self.resample_weights = weights

    def get_training_handle(self, resample=True) -> DataLoader:
        """ Returns training DataLoader, resampled every time from raw_data """
        # Filter kwargs to remove keys we pass explicitly
        dataloader_kwargs = {k: v for k, v in self.kwargs.items()
                            if k not in ('batch_size', 'shuffle')}

        if not resample:
            return _create_dataloader(
                self.raw_data['train'],
                batch_size=self.batch_size,
                shuffle=False,
                **dataloader_kwargs
            )

        # Resample the data
        resampled = _resample_np(
            self.raw_data['train'],
            sample_size=self.resample_size,
            weights=self.resample_weights,
            replace=self.resample_replace
        )

        # Apply augmentation transform if available
        transform = self.img_aug.train_transform if self.img_aug else None

        return _create_dataloader(
            resampled,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            transform=transform,
            **dataloader_kwargs
        )
