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

from efficientnet_pytorch import EfficientNet
from utils.functions import get_y_main

N_GRAPHEME = 168
N_VOWEL = 11
N_CONSONANT = 7
N_ROOT = 50


class Attention2D(nn.Module):
    def __init__(self, in_channels):
        super(Attention2D, self).__init__()
        self.dp = nn.Dropout(p=0.5)
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=1)
        self.conv2 = nn.Conv2d(512, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, features):
        attn = F.relu(self.conv1(self.dp(features)))
        attn = F.relu(self.conv2(attn))
        attn = self.sigmoid(self.conv3(attn))
        out = features * attn
        return out, attn


class Predictor(nn.Module):
    def __init__(self, model_name='efficientnet-b3', in_channels=1, middle_dim=N_ROOT, out_dims=[N_GRAPHEME, N_VOWEL, N_CONSONANT], pretrained=True, weights_path=None):

        super(Predictor, self).__init__()

        self.base_model = EfficientNet.from_pretrained(model_name, advprop=True)
        self.weights_path = weights_path

        # attention
        self.attention_grapheme = Attention2D(1536)
        self.attention_vowel = Attention2D(1536)
        self.attention_consonant = Attention2D(1536)

        # global_pool
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # 出力部付け替え
        infs = self.base_model._fc.in_features
        self.fc_grapheme_component = nn.Linear(infs, middle_dim)
        self.fc_grapheme = nn.Linear(infs+middle_dim, out_dims[0])
        self.fc_vowel = nn.Linear(infs, out_dims[1])
        self.fc_consonant = nn.Linear(infs, out_dims[2])

        if self.weights_path is not None:
            self.load_state_dict(torch.load(weight_path))
      

    def forward(self, x):

        n_batch = x.shape[0]
        inputs = torch.cat([x, x, x], dim=1)
        features = self.base_model.extract_features(inputs)

        h_grapheme, attn_grapheme = self.attention_grapheme(features)
        h_vowel, attn_vowel = self.attention_vowel(features)
        h_consonant, attn_consonant = self.attention_consonant(features)

        # GAP
        h_grapheme = self.global_pool(h_grapheme)
        h_vowel = self.global_pool(h_vowel)
        h_consonant = self.global_pool(h_consonant)
        h_grapheme = h_grapheme.view(n_batch, -1)
        h_vowel = h_vowel.view(n_batch, -1)
        h_consonant = h_consonant.view(n_batch, -1)

        # grapheme component
        component = self.fc_grapheme_component(h_grapheme)
        component = torch.sigmoid(component)

        # output
        out_grapheme = self.fc_grapheme(torch.cat([component, h_grapheme], dim=1))
        out_vowel = self.fc_vowel(h_vowel)
        out_consonant = self.fc_consonant(h_consonant)
        return torch.cat([out_grapheme, out_vowel, out_consonant, component], axis=1)



def accuracy(y, t):
    pred_label = torch.argmax(y, dim=1)
    count = pred_label.shape[0]
    correct = (pred_label == t).sum().type(torch.float32)
    acc = correct / count
    return acc


def get_y_main(y, lams):
  main_indices = [0 if l <= 0.5 else 1 for l in lams.cpu().numpy()]
  if y.ndim ==3:
    y_main = torch.cat([y[i, m, :].unsqueeze(0) for i, m in enumerate(main_indices)])
  else: 
    y_main = torch.cat([y[i, m].unsqueeze(0) for i, m in enumerate(main_indices)])

  return y_main


def calc_ohem(cross_entropy, batch_size=None, rate=0.75):
    if batch_size is None:
        return cross_entropy
    sorted_ohem_loss, idx = torch.sort(cross_entropy, descending=True)
    keep_num = min(sorted_ohem_loss.size()[0], int(batch_size*rate))
    if keep_num < sorted_ohem_loss.size()[0]:
        keep_idx_cuda = idx[:keep_num]
        cross_entropy = cross_entropy[keep_idx_cuda]
    return cross_entropy


class Classifier(nn.Module):

    def __init__(self, predictor, n_grapheme=N_GRAPHEME, n_vowel=N_VOWEL, n_consonant=N_CONSONANT, n_root=N_ROOT):
        super(Classifier, self).__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.n_root = n_root
        self.n_total_class = self.n_grapheme + self.n_vowel + self.n_consonant + self.n_root
        self.predictor = predictor

        self.metrics_keys = [
            'loss', 'loss_grapheme', 'loss_vowel', 'loss_consonant', 'loss_root',
            'acc_grapheme', 'acc_vowel', 'acc_consonant'
            ]


    def forward(self, x, y=None, lams=None):
        pred = self.predictor(x) 
        batch_size = y.shape[0]

        if isinstance(pred, tuple):
            assert len(pred) == 4
            preds = pred
        else:
            assert pred.shape[1] == self.n_total_class
            preds = torch.split(pred, [self.n_grapheme, self.n_vowel, self.n_consonant, self.n_root], dim=1)

        preds_root = preds[3] 
        y1_root = y[:, 0, 3:] 
        y2_root = y[:, 1, 3:]

        g_rate = 1.0
        v_rate = 1.0
        c_rate = 1.0
        r_rate = 0.1

        loss_root = r_rate * batch_size * (torch.mean(F.binary_cross_entropy(preds_root, y1_root.float(), reduction="none")*lams.reshape(-1,1))+torch.mean(F.binary_cross_entropy(preds_root, y2_root.float(), reduction="none")*(1-lams.reshape(-1,1))))
        loss_grapheme = g_rate * (torch.mean(calc_ohem(F.cross_entropy(preds[0], y[:, 0, 0], reduction ='none') * lams)) + torch.mean(calc_ohem(F.cross_entropy(preds[0], y[:, 1, 0], reduction ='none') * (1 - lams))))
        loss_vowel = v_rate * (torch.mean(calc_ohem(F.cross_entropy(preds[1], y[:, 0, 1], reduction ='none') * lams)) + torch.mean(calc_ohem(F.cross_entropy(preds[1], y[:, 1, 1], reduction ='none') * (1 - lams))))
        loss_consonant = c_rate * (torch.mean(calc_ohem(F.cross_entropy(preds[2], y[:, 0, 2], reduction ='none') * lams)) + torch.mean(calc_ohem(F.cross_entropy(preds[2], y[:, 1, 2], reduction ='none') * (1 - lams))))

        loss = loss_grapheme + loss_vowel + loss_consonant + loss_root 

        y_main = get_y_main(y, lams)
        metrics = {
            'loss': loss.item(),
            'loss_grapheme': loss_grapheme.item(),
            'loss_vowel': loss_vowel.item(),
            'loss_consonant': loss_consonant.item(),
            'loss_root':loss_root.item(),
            'acc_grapheme': accuracy(preds[0], y_main[:, 0]),
            'acc_vowel': accuracy(preds[1], y_main[:, 1]), 
            'acc_consonant': accuracy(preds[2], y_main[:, 2]),
        }

        return loss, metrics, pred
    
    
    def calc(self, data_loader):
        device: torch.device = next(self.parameters()).device
        self.eval()
        output_list = []
        with torch.no_grad():
            for batch in tqdm(data_loader):
                # TODO: support general preprocessing.
                # If `data` is not `Data` instance, `to` method is not supported!
                batch = batch.to(device)
                pred = self.predictor(batch)
                output_list.append(pred)
        output = torch.cat(output_list, dim=0)
        preds = torch.split(output, [self.n_grapheme, self.n_vowel, self.n_consonant], dim=1)
        return preds

    def predict_proba(self, data_loader):
        preds = self.calc(data_loader)
        return [F.softmax(p, dim=1) for p in preds]

    def predict(self, data_loader):
        preds = self.calc(data_loader)
        pred_labels = [torch.argmax(p, dim=1) for p in preds]
        return pred_labels