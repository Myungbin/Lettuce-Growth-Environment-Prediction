from helper_function import data
from helper_function import preprocessing

from ctgan import CTGAN
from table_evaluator import load_data, TableEvaluator

import pandas as pd
import numpy as np
import os, sys, glob, warnings

warnings.filterwarnings("ignore")


# load data
X_tr_list, X_te_list, y_tr_list, y_te_list = data.load_data()

# category cols
discrete_cols = ['DAT', 'obs_time']


def augmentation(epochs, file_list, save_path):
    
    # augmentation based on ctgan
    for path in file_list:

        # data
        X_tr = pd.read_csv(path)

        # preprocessing
        X_tr_pre = preprocessing.ctgan_preprocessing(X_tr)

        # fit model
        model = CTGAN(verbose=True)
        model.fit(X_tr_pre, discrete_cols, epochs=epochs)

        # generate samples based on learned model
        aug_samples = model.sample(X_tr_pre.shape[0])

        # result of sorted samples
        aug_samples.sort_values(by=['DAT', 'obs_time'], ascending=[True, True], inplace=True)

        # save aug file
        aug_samples.to_csv(f'{save_path}{path[-11:]}')


''' sample '''
# augmentation(5, X_tr_list, './data/aug_train_input/AUG_')
# augmentation(5, X_te_list, './data/aug_test_input/AUG_')
