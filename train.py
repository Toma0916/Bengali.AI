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
from model import Predictor, Classifier

from utils.dataloader import BengaliAIDataset
from utils.train_utils import create_trainer, create_evaluator, LogReport, ModelSnapshotHandler, EpochMetric

if __name__ == '__main__':

    debug = True
    device = torch.device('cuda:0')
    random_seed = 373
    batch_size = 4
    epochs = 10

    datadir = Path('.').resolve()/'data'
    outputdir = Path('.').resolve()/'output' / 'example1'
    train = pd.read_csv(datadir/'train.csv')
    cls_map = pd.read_csv(datadir/'class_map.csv')
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



    indices = [0] if debug else [0, 1, 2, 3]
    train_images = prepare_image(datadir, data_type='train', indices=indices)
    n_dataset = train_images.shape[0]

    train_data_size = 200 if debug else int(n_dataset * 0.9)
    valid_data_size = 100 if debug else int(n_dataset - train_data_size)
    perm = np.random.RandomState(random_seed).permutation(n_dataset)
    train_dataset = BengaliAIDataset(train_images, train_labels, indices=perm[:train_data_size])
    valid_dataset = BengaliAIDataset(train_images, train_labels, indices=perm[train_data_size:train_data_size+valid_data_size])

    predictor = Predictor(weights_path=None)
    classifier = Classifier(predictor).to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-10)

    trainer = create_trainer(classifier, optimizer, device)
    evaluator = create_evaluator(classifier, device)

    EpochMetric().attach(trainer, 'recall')
    EpochMetric().attach(evaluator, 'recall')
    
    pbar = ProgressBar()
    pbar.attach(trainer, metric_names='all')


    def run_evaluator(engine):
        evaluator.run(valid_loader)

    
    def schedule_lr(engine):
        metrics = evaluator.state.metrics
        avg_mae = metrics['loss']

        # --- update lr ---
        lr = scheduler.optimizer.param_groups[0]['lr']
        scheduler.step(avg_mae)
        log_report.report('lr', lr)
    
    os.makedirs(outputdir, exist_ok=True)
    log_report = LogReport(evaluator, outputdir)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, run_evaluator)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, schedule_lr)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_report)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, ModelSnapshotHandler(predictor, filepath=outputdir/'predictor_{count:06}.pt'))

    # save_configs()
    trainer.run(train_loader, max_epochs=epochs)