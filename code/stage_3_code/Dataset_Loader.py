'''
Concrete IO class for the stage 3 image datasets (MNIST, ORL, CIFAR)
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import pickle
import numpy as np


class Dataset_Loader(dataset):
    '''
    Each provided file (MNIST / ORL / CIFAR) is a pickled dict:
        {'train': [{'image': ndarray, 'label': int}, ...],
         'test':  [{'image': ndarray, 'label': int}, ...]}
    Image shapes: MNIST 28x28 (gray), ORL 112x92x3 (gray stored as 3 equal channels),
    CIFAR 32x32x3 (color).

    load() returns tensors-ready numpy arrays in PyTorch (N, C, H, W) layout:
        {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
    '''
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    # if True, keep only one channel of a 3-channel gray image (ORL)
    use_single_channel = False
    # per-channel standardization using training-set mean/std
    normalize = True

    def __init__(self, dName=None, dDescription=None, use_single_channel=False, normalize=True):
        super().__init__(dName, dDescription)
        self.use_single_channel = use_single_channel
        self.normalize = normalize

    def _to_array(self, instances):
        X = np.array([np.asarray(inst['image'], dtype=np.float32) for inst in instances])
        y = np.array([int(inst['label']) for inst in instances], dtype=np.int64)
        if X.ndim == 3:                     # (N, H, W) -> (N, 1, H, W)
            X = X[:, None, :, :]
        else:                               # (N, H, W, C) -> (N, C, H, W)
            X = X.transpose(0, 3, 1, 2)
            if self.use_single_channel:     # R = G = B for ORL, so one channel suffices
                X = X[:, :1, :, :]
        return X, y

    def load(self):
        print('loading data...')
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)
        f.close()

        X_train, y_train = self._to_array(data['train'])
        X_test, y_test = self._to_array(data['test'])

        # pixel values are 0-255 -> scale to [0, 1]
        if X_train.max() > 1.0:
            X_train /= 255.0
            X_test /= 255.0

        if self.normalize:
            mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
            std = X_train.std(axis=(0, 2, 3), keepdims=True) + 1e-8
            X_train = (X_train - mean) / std
            X_test = (X_test - mean) / std

        # ORL labels are 1..40; remap any label set to 0..K-1 for CrossEntropyLoss
        classes = np.unique(np.concatenate([y_train, y_test]))
        label_map = {c: i for i, c in enumerate(classes)}
        y_train = np.array([label_map[c] for c in y_train], dtype=np.int64)
        y_test = np.array([label_map[c] for c in y_test], dtype=np.int64)

        print('train:', X_train.shape, 'test:', X_test.shape, 'classes:', len(classes))
        return {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
