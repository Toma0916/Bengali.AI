import argparse
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
warnings.filterwarnings('ignore')

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
torch.backends.cudnn.benchmark=True

from sklearn import preprocessing
from sklearn.model_selection import KFold
import sklearn.metrics

from ignite.engine.engine import Engine, Events
from ignite.metrics import Average
from ignite.metrics.metric import Metric
from ignite.contrib.handlers import ProgressBar

import cv2

from model import Predictor, Classifier

from utils.dataloader import BengaliAIDataset, Transform
from utils.train_utils import create_trainer, create_evaluator, LogReport, ModelSnapshotHandler, EpochMetric
from utils.functions import prepare_image, get_train_labels, save_configs

if __name__ == '__main__':



    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    # parser.add_argument('--device')
    parser.add_argument('--output_dir', default='default')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--mixup_ratio', type=float, default=0.)
    parser.add_argument('--cutmix_ratio', type=float, default=0.)
    parser.add_argument('--affine_ratio', type=float, default=0.)

    args = parser.parse_args()

    debug = args.debug
    device = torch.device('cuda:0')
    random_seed = args.seed
    batch_size = args.batch_size
    epochs = args.epochs
    datadir = Path('.').resolve()/'data'
    outputdir = Path('.').resolve()/'output' / args.output_dir
    train = pd.read_csv(datadir/'train.csv')
    cls_map = pd.read_csv(datadir/'class_map.csv')
    train_labels = get_train_labels(train, cls_map)
    writer = SummaryWriter(log_dir=outputdir)

    train_config = {
        'debug': debug,
        'random_seed': random_seed,
        'batch_size': batch_size,
        'epochs': epochs,
    }

    train_aug_config = {
        'mixup_ratio': args.mixup_ratio,
        'cutmix_ratio': args.cutmix_ratio,
        'affine_ratio': args.affine_ratio,
        'normalize': True,
    }

    valid_aug_config = {
        'mixup_ratio': 0.,
        'cutmix_ratio': 0.,
        'affine_ratio': 1.,
        'normalize': True,
    }

    config = train_config
    for key, value in train_aug_config.items():
        config['train' + key] = value
    for key, value in valid_aug_config.items():
        config['valid_' + key] = value

    indices = [0] if debug else [0, 1, 2, 3]
    train_images = prepare_image(datadir, data_type='train', indices=indices)
    n_dataset = train_images.shape[0]

    train_data_size = 200 if debug else int(n_dataset * 0.9)
    valid_data_size = 100 if debug else int(n_dataset - train_data_size)
    perm = np.random.RandomState(random_seed).permutation(n_dataset)
    train_dataset = BengaliAIDataset(train_images, train_labels, transform=Transform(train_aug_config), indices=perm[:train_data_size])
    valid_dataset = BengaliAIDataset(train_images, train_labels, transform=Transform(valid_aug_config), indices=perm[train_data_size:train_data_size+valid_data_size])

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
        if not isinstance(avg_mae, float):
            avg_mae = avg_mae.mean()
        # --- update lr ---
        lr = scheduler.optimizer.param_groups[0]['lr']
        scheduler.step(avg_mae)
        log_report.report('lr', lr)
    
    os.makedirs(outputdir, exist_ok=True)
    log_report = LogReport(evaluator, outputdir, debug=debug, logger=writer)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, run_evaluator)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, schedule_lr)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_report)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, ModelSnapshotHandler(predictor, filepath=outputdir/'predictor_{count:06}.pt'))

    save_configs(config, outputdir)

    trainer.run(train_loader, max_epochs=epochs)
    writer.close()
