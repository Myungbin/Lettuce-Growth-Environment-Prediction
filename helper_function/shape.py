import pandas as pd
import numpy as np

def linear(df):

    ''' del '''
    # df = pd.read_csv(df)
    ''' del '''

    # day, obstime range
    day = range(0, 28)
    obstime = range(0, 24)

    # linear columns
    cols = ['내부온도관측치', '내부습도관측치', 'co2관측치', 'ec관측치', '시간당분무량', '시간당백색광량', '시간당적색광량', '시간당청색광량']
    linear_cols = [(col + str(h)) for col in cols for h in obstime]

    # dataframe to return
    result = pd.DataFrame(columns=linear_cols)

    # linear
    for d in day: # 0 ~ 28
        df_day = df.iloc[(24 * d):((24 * d) + 24)]
        append_vals = []

        for h in obstime: # 0 ~ 24
            for col in cols:
                val = df_day.iloc[h, :][col]
                append_vals.append(val)

        # print(append_vals)
        target = pd.DataFrame(columns=linear_cols, data=[append_vals])
        result = pd.concat([result, target], axis=0)

    return result


def groupby_day(mode, path, file_list):

    # add day column
    df_day = pd.DataFrame(columns=['day'], data=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27])

    # return df
    result = pd.DataFrame()

    # linear df
    for f in file_list:

        file_path = path + f
        df = pd.read_csv(file_path)

        # concat with day column
        df = pd.concat([df_day, df], axis=1)

        # concat with return df
        result = pd.concat([result, df], axis=0)

    if mode == 'train':
        result.to_csv(r'C:\Project\dacon-lettuce-growth\Lettuce-Growth-Environment-Prediction\data\aug_input\train\3_groupby_day\TRAIN.csv', index=False)
    if mode == 'test':
        result.to_csv(r'C:\Project\dacon-lettuce-growth\Lettuce-Growth-Environment-Prediction\data\aug_input\test\3_groupby_day\TEST.csv', index=False)

    return result


def get_groups(mode, day):

    if mode == 'train':
        df = pd.read_csv('./data/aug_input/train/3_groupby_day/TRAIN.csv')
    if mode == 'test':
        df = pd.read_csv('./data/aug_input/test/3_groupby_day/TEST.csv')
    
    groups = df.groupby('day')
    group = groups.get_group(day)

    return group


''' sample '''
''' 
linear(r'C:\Project\dacon-lettuce-growth\Lettuce-Growth-Environment-Prediction\data\aug_input\train\1_del_cumsum\TRAIN0.csv') '''
'''
from os import listdir
linear_path = '../data/aug_input/train/2_linear/'
linear_files = listdir(linear_path)
groupby_day(linear_path, linear_files) '''
''' 
get_groups('train', 0) '''