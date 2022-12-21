from helper_function import dataset

import pandas as pd
import numpy as np


# load data
X_tr_aug_list, X_te_aug_list = dataset.load_data_aug()

# cols
cumsum_cols = ['일간누적분무량', '일간누적백색광량', '일간누적적색광량', '일간누적청색광량', '일간누적총광량']
sum_cols = ['시간당총광량']
target_cols = ['시간당분무량', '시간당백색광량', '시간당적색광량', '시간당청색광량']


def make_cumsum_cols(df):

    # data
    X = df

    # create sum_cols
    X['시간당총광량'] = X['시간당백색광량'] + X['시간당적색광량'] + X['시간당청색광량']

    # create cumsum_cols
    for idx, col in enumerate(cumsum_cols):
        if col != '일간누적총광량':
            X[col] = X[target_cols[idx]].cumsum()
        else:
            X[col] = X['시간당총광량'].cumsum()

    return X


''' sample '''
# make_cumsum_cols(df)
# make_cumsum_cols(df)