import pandas as pd
import numpy as np
import datetime

def cumsum_group_max(df):
    '''
    *누적값은 media이 아닌 일별 최대값이 들어가야함
    '''
    df_a = pd.DataFrame()

    for i, v in enumerate(df["Case"].unique()):
        train_old = df[df['Case']==v]
        train_old = train_old.groupby(['DAT']).max().reset_index()
        df_a = pd.concat([df_a, train_old])
    return df_a


def group_median(df):  
    '''
    변수의 Case별로 DAT을 기준으로 groupby 하여 대표값 설정
    '''
    df_a = pd.DataFrame()

    for i, v in enumerate(df["Case"].unique()):
        train_old = df[df['Case']==v]
        train_old = train_old.groupby(['DAT']).mean().reset_index()
        df_a = pd.concat([df_a, train_old])
    return df_a


def concat_df(df, cumsum_df):
    '''
    누적값을 원래 데이터와 concat
    '''
    cumsum_list = ['일간누적분무량', '일간누적백색광량', '일간누적적색광량', '일간누적청색광량', '일간누적총광량']
    for col in cumsum_list:
        df[col] = cumsum_df[col]
        
    return df


def diff_temp(train, test):
    '''
    일교차
    '''
    train_diff_temp = []
    for i in range(784):
        temp_train = train[i*24:(i+1)*24]
        diff_train = temp_train['내부온도관측치'].max() - temp_train['내부온도관측치'].min()
        train_diff_temp.append(diff_train)

    test_diff_temp = []
    for i in range(140):
        temp_test = test[i*24:(i+1)*24]
        diff_test = temp_test['내부온도관측치'].max() - temp_test['내부온도관측치'].min()
        test_diff_temp.append(diff_test)

    return train_diff_temp, test_diff_temp


def none_light(train, test):
    '''
    광합성을 못하는 시간
    '''
    train_night = []
    for i in range(784):
        nigth_train = train[i*24:(i+1)*24]
        train_night.append(nigth_train[nigth_train['시간당백색광량']==0]['시간당백색광량'].count())
    
    test_night = []
    for i in range(140):
        nigth_test = test[i*24:(i+1)*24]
        test_night.append(nigth_test[nigth_test['시간당백색광량']==0]['시간당백색광량'].count())
    
    return train_night, test_night


def water(train, test):
    '''
    하루에 물을 주는 횟수
    '''
    train_water = []
    for i in range(784):
        water_train = train[i*24:(i+1)*24]
        train_water.append(water_train[water_train['시간당분무량']!=0]['시간당분무량'].count())
    
    test_water = []
    for i in range(140):
        water_test = test[i*24:(i+1)*24]
        test_water.append(water_test[water_test['시간당분무량']!=0]['시간당분무량'].count())
    
    return train_water, test_water


def accumulate(train, test, col):
    '''
    총 누적합
    '''
    train['월간'+col] = 0
    for i in range(28):
        result = (train['일간'+col][i*28:(i+1)*28].cumsum())
        train['월간'+col][i*28:(i+1)*28] = result


    test['월간'+col] = 0
    for i in range(5):
        result = (test['일간'+col][i*28:(i+1)*28].cumsum())
        test["월간"+col][i*28:(i+1)*28] = result