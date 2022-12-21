import pandas as pd
import numpy as np
import os

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


def create(mode):

    if mode == 'train':

        # 0 ~ 27 일의 조합들 생성
        aug_path = './data/aug_input/train/4_aug/'
        aug_files = os.listdir(aug_path)
        linear_cols = pd.read_csv(aug_path + aug_files[0]).columns.tolist()

        ''' have to modify '''

        for idx in range(0, 100):
            
            result = pd.DataFrame(columns=linear_cols)

            for aug_file in aug_files:
                file_path = aug_path + aug_file

                target_vals = pd.read_csv(file_path).iloc[idx].tolist()        # samples of each day (type : list)
                target = pd.DataFrame(columns=linear_cols, data=[target_vals]) # samples of each day (type : dataframe)
                result = pd.concat([result, target], axis=0)                   # concat with other day

            result.sort_values(by=['day'], inplace=True)
            result.to_csv(f'./data/aug_input/train/5_create/TRAIN{idx}.csv', index=False)

        ''' have to modify '''
   

    elif mode == 'test':

        return


def reshape(mode):

    if mode == 'train':

        raw_cols = ['내부온도관측치', '내부습도관측치', 'co2관측치', 'ec관측치', '시간당분무량', '시간당백색광량', '시간당적색광량', '시간당청색광량']
        linear_cols = pd.read_csv('./data/aug_input/train/5_create/TRAIN0.csv').columns.tolist()

        concat_path = './data/aug_input/train/5_create/'
        concat_files = os.listdir(concat_path)
        
        for f_idx, concat_file in enumerate(concat_files):
            file_path = concat_path + concat_file

            target = pd.read_csv(file_path) # = case sample
            result = pd.DataFrame(columns=raw_cols)
            eachtime = pd.DataFrame(columns=raw_cols)
            
            for idx in range(len(target)):            # = day0 ~ day27
                for i, r_col in enumerate(raw_cols):  # = dayn time

                    vals = target.iloc[i, (i * 24 + 1):(i * 24 + 25)].tolist()
                    eachtime[r_col] = vals

                result = pd.concat([result, eachtime], axis=0)
            
            result.to_csv(f'./data/aug_input/train/6_reshape/TRAIN{f_idx}', index=False)

    if mode == 'test':

        return


''' sample '''
''' 2) '''
# linear(r'C:\Project\dacon-lettuce-growth\Lettuce-Growth-Environment-Prediction\data\aug_input\train\1_del_cumsum\TRAIN0.csv')
''' 3) '''
# from os import listdir
# linear_path = '../data/aug_input/train/2_linear/'
# linear_files = listdir(linear_path)
# groupby_day(linear_path, linear_files)
''' 4) '''
# get_groups('train', 0)
''' 5) '''
# create('train')
''' 6) '''
# reshape('train')