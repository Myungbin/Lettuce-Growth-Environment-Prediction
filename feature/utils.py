import numpy as np
import pandas as pd
from feature.base_dataset import limit_range, time_value
from glob import glob


def log_scale(train, test):
    """range가 큰 값을 로그변환(월간누적 데이터만 적용)

    Args:
        train (DataFrame): train data set
        test (DataFrame): test data set
    """
    log_col_list = ['월간누적분무량', '월간누적백색광량', '월간누적청색광량', '월간누적적색광량']
    for col in log_col_list:
        train[col] = np.log1p(train[col])
        test[col] = np.log1p(test[col])


def transpose_data(input_path):
    """data transpose

    Args:
        input_path (str): path

    Returns:
        DataFrame
    """    
    input_list = sorted(glob(input_path))
    df_t = pd.DataFrame()
    for i in input_list:
        df = pd.read_csv(i)
        df = time_value(df)
        df = limit_range(df)
        df = df.fillna(method='ffill')
        df = df.drop(['시간당분무량', '시간당백색광량',
                      '시간당적색광량', '시간당청색광량', '시간당총광량',
                      '일간누적청색광량', '일간누적적색광량', '일간누적백색광량'], axis=1)

        df = pd.pivot_table(df, index=['DAT'], columns=['obs_time'])
        df.columns = [''.join(str(col)) for col in df.columns]
        df = df.reset_index()

        df_t = pd.concat([df_t, df])
    df_t = df_t.drop(['DAT'], axis=1)
    return df_t

