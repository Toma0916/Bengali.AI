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