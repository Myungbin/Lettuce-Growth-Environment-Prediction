from helper_function import dataset
from helper_function import preprocessing
from helper_function import shape
import cumsum

from ctgan import CTGAN
from table_evaluator import load_data, TableEvaluator

from os import listdir
import pandas as pd
import numpy as np
import os, sys, glob, warnings

warnings.filterwarnings("ignore")


# load data
X_tr_list, X_te_list, y_tr_list, y_te_list = dataset.load_data()

# category cols
discrete_cols = ['DAT', 'obs_time']


def augmentation(mode, epochs, file_list, save_path):

    '''
    
    # augmentation based on ctgan
    for idx, path in enumerate(file_list):

        # data
        X = pd.read_csv(path)

        # del cumsum cols
        X = preprocessing.ctgan_preprocessing(X)
        X_pre = X.iloc[:, 2:] # except discrete_cols
        if mode == 'train':
            X_pre.to_csv(f'./data/aug_input/train/1_del_cumsum/TRAIN{idx}.csv', index=False)
        elif mode == 'test':
            X_pre.to_csv(f'./data/aug_input/test/1_del_cumsum/TEST{idx}.csv', index=False)

        # linear
        X_pre = shape.linear(X_pre)
        if mode == 'train':
            X_pre.to_csv(f'./data/aug_input/train/2_linear/TRAIN{idx}.csv', index=False)
        elif mode == 'test':
            X_pre.to_csv(f'./data/aug_input/test/2_linear/TEST{idx}.csv', index=False)

    # groupby day
    if mode == 'train':
        linear_path = './data/aug_input/train/2_linear/'
        linear_files = listdir(linear_path)
        X_pre = shape.groupby_day('train', linear_path, linear_files)
    elif mode == 'test':
        linear_path = './data/aug_input/test/2_linear/'
        linear_files = listdir(linear_path)
        X_pre = shape.groupby_day('test', linear_path, linear_files)
    
    # augmentation based on ctgan
    if mode == 'train':

        for d in range(0, 28): # train
            X_pre = shape.get_groups('train', d) # X_pre.shape = (28, 193)

            # fit model
            model = CTGAN(verbose=True)
            model.fit(X_pre, epochs=epochs) # discrete_cols, epochs=epochs)

            # generate samples based on learned model
            aug_X = model.sample(100)
            aug_X.to_csv(f'./data/aug_input/train/4_aug/TRAIN{d}', index=False)

    elif mode == 'test': 

        for d in range(0, 28):  # test
            X_pre = shape.get_groups('test', d)

            # fit model
            model = CTGAN(verbose=True)
            model.fit(X_pre, epochs=epochs) # discrete_cols, epochs=epochs)

            # generate samples based on learned model
            aug_X = model.sample(100)
            aug_X.to_csv(f'./data/aug_input/test/4_aug/TEST{d}', index=False)

    '''

    # create dataset
    DAT_OBS_TIME = pd.read_csv('./data/train_input/CASE_01.csv').iloc[:, :2]

    if mode == 'train':

        '''
        # create
        shape.create('train')
        # reshape
        shape.reshape('train')
        '''
        # preprocessing to input predict model
        for idx, f in enumerate(os.listdir('./data/aug_input/train/6_reshape/')):
            file_path = './data/aug_input/train/6_reshape/' + f   # file path

            target = pd.read_csv(file_path)                       # dataframe
            target = preprocessing.save_preprocessing(target)     # range cutoff
            target = cumsum.make_cumsum_cols(target)              # create cumsum cols
            target = pd.concat([DAT_OBS_TIME, target], axis=1)    # create DAT, obs_time cols
            
            target.to_csv(f'./data/aug_input/train/7_fin/TRAIN{idx}.csv', index=False) # save file

    if mode == 'test':

        # create

        return
        


''' sample '''
augmentation('train', 50, X_tr_list, './data/aug_train_input/AUG_')
# augmentation('test', 50, X_te_list, './data/aug_test_input/AUG_')