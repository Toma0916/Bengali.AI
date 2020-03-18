import os
from pathlib import Path
import random
import sys
import gc
import six
import json
from logging import getLogger
from time import perf_counter
import warnings
import glob

import numpy as np 
from numpy.random.mtrand import RandomState
import pandas as pd

from PIL import Image, ImageEnhance, ImageOps

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn import preprocessing
from sklearn.model_selection import KFold
import sklearn.metrics

from ignite.engine.engine import Engine, Events
from ignite.metrics import Average
from ignite.metrics.metric import Metric
from ignite.contrib.handlers import ProgressBar

import cv2


class DatasetMixin(Dataset):

    def __init__(self, transform=None):
        self.transform = transform

    def __getitem__(self, index):
        """Returns an example or a sequence of examples."""
        if torch.is_tensor(index):
            index = index.tolist()
        if isinstance(index, slice):
            current, stop, step = index.indices(len(self))
            return [self.get_example_wrapper(i) for i in
                    six.moves.range(current, stop, step)]
        elif isinstance(index, list) or isinstance(index, np.ndarray):
            return [self.get_example_wrapper(i) for i in index]
        else:
            return self.get_example_wrapper(index)

    def __len__(self):
        """Returns the number of data points."""
        raise NotImplementedError
      
    def get_example_wrapper(self, i):
        """Wrapper of `get_example`, to apply `transform` if necessary"""
        example = self.get_example(i)
        if self.transform:
            example = self.transform(example, self)  # nyan: TransformにDatasetを参照させたいのでselfを渡す
        return example

    def get_example(self, i):
        """Returns the i-th example.

        Implementations should override it. It should raise :class:`IndexError`
        if the index is invalid.

        Args:
            i (int): The index of the example.

        Returns:
            The i-th example.

        """
        raise NotImplementedError
  

class BengaliAIDataset(DatasetMixin):
    def __init__(self, images, labels=None, transform=None, indices=None):
        super(BengaliAIDataset, self).__init__(transform=transform)
        self.images = images
        self.labels = labels
        if indices is None:
            indices = np.arange(len(images))
        self.indices = indices
        self.train = labels is not None

    def __len__(self):
        """return length of this dataset"""
        return len(self.indices)

    def get_example(self, i):
        """Return i-th data"""
        i = self.indices[i]
        x = self.images[i]
        
        x = (x.astype(np.float32) - 0.0692) / 0.2051
        x = np.expand_dims(x, axis=0)

        # Opposite white and black: background will be white and
        # for future Affine transformation
        if self.train:
            y = self.labels[i]
            return x, y
        else:
            return x
