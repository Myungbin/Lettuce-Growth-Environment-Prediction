from helper_function import dataset

import pandas as pd
import numpy as np


# load data
X_tr_aug_list, X_te_aug_list = dataset.load_data_aug()

# cols
cumsum_cols = ['일간누적분무량', '일간누적백색광량', '일간누적적색광량', '일간누적청색광량', '일간누적총광량']
sum_cols = ['시간당총광량']
target_cols = ['시간당분무량', '시간당백색광량', '시간당적색광량', '시간당청색광량']


def make_cumsum_cols(file_list, save_path):

    for path in file_list:

        # data
        X = pd.read_csv(path)

        # create sum_cols
        X['시간당총광량'] = X['시간당백색광량'] + X['시간당적색광량'] + X['시간당청색광량']

        # create cumsum_cols
        for idx, col in enumerate(cumsum_cols):
            if col != '일간누적총광량':
                X[col] = X[target_cols[idx]].cumsum()
            else:
                X[col] = X['시간당총광량'].cumsum()

        # save df
        X.to_csv(f'{save_path}{path[-11:]}', index=False)

    return


''' sample '''
make_cumsum_cols(X_tr_aug_list, './data/aug_cumsum_train_input/AUG_')
make_cumsum_cols(X_te_aug_list, './data/aug_cumsum_test_input/AUG_')