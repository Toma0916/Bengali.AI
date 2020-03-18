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

from prepare import prepare_image
from model import EfficientNet_b3, Classifier

from utils.dataloader import BengaliAIDataset


if __name__ == '__main__':

    debug = True
    device = torch.device('cuda:0')


    datadir = Path('.').resolve()/'data'
    train = pd.read_csv(datadir/'train.csv')
    train_labels = train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values
    indices = [0] if debug else [0, 1, 2, 3]
    train_images = prepare_image(datadir, data_type='train', indices=indices)


    train_dataset_sample = BengaliAIDataset(train_images, train_labels)

    predictor = EfficientNet_b3()
    classifier = Classifier(predictor).to(device)


    import pdb; pdb.set_trace()
