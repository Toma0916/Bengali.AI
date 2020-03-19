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


def get_y_main(y, lams):
    main_indices = [0 if l <= 0.5 else 1 for l in lams.cpu().numpy()]
    if y.ndim ==3:
      y_main = torch.cat([y[i, m, :].unsqueeze(0) for i, m in enumerate(main_indices)])
    else: 
      y_main = torch.cat([y[i, m].unsqueeze(0) for i, m in enumerate(main_indices)])  
    return y_main


def prepare_image(datadir, data_type='train', indices=[0, 1, 2, 3]):

    assert data_type in ['train', 'test']
    image_df_list = [pd.read_parquet(datadir / f'{data_type}_image_data_{i}.parquet') for i in indices]

    print('image_df_list', len(image_df_list))
    HEIGHT = 137
    WIDTH = 236
    images = [df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH) for df in image_df_list]
    del image_df_list
    gc.collect()
    images = np.concatenate(images, axis=0)
    return images


def get_train_labels(train, cls_map):
    
    grapheme_map = cls_map.loc[cls_map['component_type']=='grapheme_root', :]

    grapheme_set = []
    for i, row in grapheme_map.iterrows():
        grapheme_set.extend(list(row['component']))
    grapheme_set = sorted(list(set(grapheme_set)))

    graphemeroot2num = {}
    for i, graphemeroot in enumerate(grapheme_set):
        graphemeroot2num[graphemeroot] = i

    graphemeroot2component = {}
    for i, row in grapheme_map.iterrows():
        item = [0] * 50
        for cmp in list(row['component']):
            item[graphemeroot2num[cmp]] = 1
        graphemeroot2component[row['label']] = np.array(item)

    train_labels = train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values
    train_labels__ = np.zeros((train_labels.shape[0], 53), dtype=np.int64)
    for i in range(train_labels.shape[0]):
        label = np.concatenate([train_labels[i], graphemeroot2component[train_labels[i][0]]])
        train_labels__[i] = label

    train_labels = train_labels__

    return train_labels


def save_configs(config, outputdir):

  with open(outputdir/'train_config.json', 'w') as f:
    json.dump(config, f)

