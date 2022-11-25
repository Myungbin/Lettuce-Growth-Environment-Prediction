import random
import pandas as pd
import numpy as np
import os
from glob import glob

train_input_list = sorted(glob('./data/train_input/*.csv'))
train_target_list = sorted(glob('./data/train_target/*.csv'))

test_input_list = sorted(glob('./data/test_input/*.csv'))
test_target_list = sorted(glob('./data/test_target/*.csv'))

def make_dataset(all_input_list, all_target_list):
    df_all = pd.DataFrame()
    length = len(all_input_list)
    for idx in range(length):
        X = pd.read_csv(all_input_list[idx])
        y = pd.read_csv(all_target_list[idx])
        y['DAT'] = y['DAT']-1
        df_concat = pd.merge(X, y, on='DAT', how='left')
        df_concat['Case'] = idx+1
        df_all = pd.concat([df_all, df_concat])
    return df_all

train = make_dataset(train_input_list, train_target_list)
test = make_dataset(test_input_list, test_target_list)