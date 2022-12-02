import numpy as np
import pandas as pd

def log_scale(train, test):
    '''
    range가 큰 값을 로그변환 => 월간누적 데이터만 적용
    return: log변환된 train, test data
    '''
    log_col_list = ['월간누적분무량', '월간누적백색광량', '월간누적청색광량', '월간누적적색광량']
    for col in log_col_list:
        train[col] = np.log1p(train[col])
        test[col] = np.log1p(test[col])
        
        
def basic_stat():
    train_1 = pd.read_csv('./data/clip_preprocessing/clip_train_quantile0.25.csv')
    test_1 = pd.read_csv('./data/clip_preprocessing/clip_test_quantile0.25.csv')
    train_3 = pd.read_csv('./data/clip_preprocessing/clip_train_quantile0.75.csv')
    test_3 = pd.read_csv('./data/clip_preprocessing/clip_test_quantile0.75.csv')
    train_std = pd.read_csv('./data/clip_preprocessing/clip_train_std.csv')
    test_std = pd.read_csv('./data/clip_preprocessing/clip_test_std.csv')
    train_mid = pd.read_csv('./data/clip_preprocessing/clip_train_median.csv')
    test_mid = pd.read_csv('./data/clip_preprocessing/clip_test_median.csv')
    train_sem = pd.read_csv('./data/clip_preprocessing/clip_train_sem.csv')
    test_sem = pd.read_csv('./data/clip_preprocessing/clip_test_sem.csv')
    
    train_3.columns = [col+' q3' for col in train_3.columns]
    test_3.columns = [col+' q3' for col in test_3.columns]
    train_1.columns = [col+' q1' for col in train_1.columns]
    test_1.columns = [col+' q1' for col in test_1.columns]
    train_std.columns = [col+' std' for col in train_std.columns]
    test_std.columns = [col+' std' for col in test_std.columns]
    train_mid.columns = [col+' mid' for col in train_mid.columns]
    test_mid.columns = [col+' mid' for col in test_mid.columns]
    train_sem.columns = [col+' sem' for col in train_sem.columns]
    test_sem.columns = [col+' sem' for col in test_sem.columns]
    
    train = train.reset_index()
    test  = test.reset_index()
    train_3 = train_3.reset_index()
    test_3 = test_3.reset_index()
    train_1 = train_1.reset_index()
    test_1 = test_1.reset_index()
    train_std = train_std.reset_index()
    test_std = test_std.reset_index()
    train_mid = train_mid.reset_index()
    test_mid = test_mid.reset_index()
    train_sem = train_sem.reset_index()
    test_sem = test_sem.reset_index()