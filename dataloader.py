import random
import pandas as pd
import numpy as np
import os
import glob

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm

import warnings
warnings.filterwarnings(action='ignore') 
all_input_list = sorted(glob.glob('./data/train_input/*.csv'))
all_target_list = sorted(glob.glob('./data/train_target/*.csv'))

train_input_list = all_input_list[:25]
train_target_list = all_target_list[:25]

val_input_list = all_input_list[25:]
val_target_list = all_target_list[25:]


input_paths = train_input_list
target_paths = train_target_list
data_list = []
label_list = []

for input_path, target_path in tqdm(zip(input_paths, target_paths)):
    input_df = pd.read_csv(input_path)
    target_df = pd.read_csv(target_path)
    
    input_df = input_df.drop(columns=['obs_time'])
    input_df = input_df.fillna(method='ffill')
    
    input_length = int(len(input_df)/24)
    target_length = int(len(target_df))
    
    for idx in range(target_length):
        time_series = input_df[24*idx:24*(idx+1)].values
        data_list.append(torch.Tensor(time_series))
    for label in target_df["predicted_weight_g"]:
        label_list.append(label)

print(data_list)
print(label_list)
print(len(data_list))
print(len(label_list))
