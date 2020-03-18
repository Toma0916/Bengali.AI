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
import pandas as pd



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
