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
from skimage.transform import AffineTransform, warp
import sklearn.metrics

from ignite.engine.engine import Engine, Events
from ignite.metrics import Average
from ignite.metrics.metric import Metric
from ignite.contrib.handlers import ProgressBar

import cv2

HEIGHT = 137
WIDTH = 236


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
        # Opposite white and black: background will be white and
        # for future Affine transformation
        if self.train:
            y = self.labels[i]
            return x, y
        else:
            return x


def _evaluate_ratio(ratio):
    if ratio <= 0.:
        return False
    return np.random.uniform() < ratio


def affine_image(img):
    """

    Args:
        img: (h, w) or (1, h, w)

    Returns:
        img: (h, w)
    """
    if img.ndim == 3:
        img = img[0]

    # --- scale ---
    min_scale = 0.9
    max_scale = 1.2
    sx = np.random.uniform(min_scale, max_scale)
    sy = np.random.uniform(min_scale, max_scale)

    # --- rotation ---
    max_rot_angle = 7
    rot_angle = np.random.uniform(-max_rot_angle, max_rot_angle) * np.pi / 180.

    # --- shear ---
    max_shear_angle = 10
    shear_angle = np.random.uniform(-max_shear_angle, max_shear_angle) * np.pi / 180.

    # --- translation ---
    max_translation = 4
    tx = np.random.randint(-max_translation, max_translation)
    ty = np.random.randint(-max_translation, max_translation)

    tform = AffineTransform(scale=(sx, sy), rotation=rot_angle, shear=shear_angle,
                            translation=(tx, ty))
    transformed_image = warp(img, tform)
    assert transformed_image.ndim == 2
    return transformed_image


def rand_bbox(size, lam):
    W, H = size[-2:]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W//2 - W//4, W//2 + W//4)
    cy = np.random.randint(H//2 - H//4, H//2 + H//4)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    if bbx1 == 0:
      bbx2 = np.clip(bbx2, W // 6, None)
    elif bbx2 == W:
      bbx1 = np.clip(bbx1, None, W - W // 6)
    
    if bby1 == 0:
      bby2 = np.clip(bby2, H // 6, None)
    elif bby2 == H:
      bby1 = np.clip(bby1, None, H - H // 6)
   
    return bbx1, bby1, bbx2, bby2


class Transform:

  def __init__(self, aug_config):
    
    self.mixup_ratio = aug_config['mixup_ratio']
    self.cutmix_ratio = aug_config['cutmix_ratio']
    self.affine_ratio = aug_config['affine_ratio']
    self.normalize = aug_config['normalize']


  def __call__(self, example, dataset):

    is_train = True

    if isinstance(example, tuple):
      img = example[0]
      label = example[1]
    else: 
      img = example
      is_train = False
      
    img = (255 - img)
    img = (img*(255.0/img.max())).astype(np.uint8)
    img = img.astype(np.float32) / 255

    r = np.random.rand()
    if 0. <= r < self.mixup_ratio + self.cutmix_ratio :
      img2, label2 = dataset.get_example(np.random.randint(len(dataset)))
      img2= (255 - img2)
      img2 = (img2*(255.0/img2.max())).astype(np.uint8)
      img2 = img2.astype(np.float32) / 255

      mixup_alpha = 0.5  
      cutmix_alpha = 0.8 

      if r < self.mixup_ratio:  
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        img = img * lam + img2 * (1 - lam)
        label = [np.array([label, label2]), lam]
      else:  # nyan: cutmix
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        bbx1, bby1, bbx2, bby2 = rand_bbox(img.shape, lam)
        img[bbx1:bbx2, bby1:bby2] = img2[bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (np.product(img.shape)))
        label = [np.array([label, label2]), lam]

    else:
      label = [np.array([label, label]), 0.5]
  
    if _evaluate_ratio(self.affine_ratio):
        img = affine_image(img)

    if self.normalize:
        img = (img.astype(np.float32) - 0.0692) / 0.2051
    
    assert img.ndim == 2
    img = np.expand_dims(img, axis=0)
    
    return img, label if is_train else img
    

