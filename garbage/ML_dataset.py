import random
import pandas as pd
import numpy as np
import os
import glob
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def make_dataset(input_path, target_path):
    train = pd.DataFrame()
    all_input_list = sorted(glob.glob(input_path))
    all_target_list = sorted(glob.glob(target_path))
    for x, y in zip(all_input_list,all_target_list):
        x = pd.read_csv(x)
        y = pd.read_csv(y)
        x['obs_time'] = x['obs_time'].str[:2]
        x = x.iloc[:, 1:]
        col_list = x.columns[1:]
        for i in range(0,28) :
            day = x.iloc[24*i:24*i+24]
            time_list = day['obs_time'].unique()
            if len(time_list) > 24 :
                for i in range(0,len(time_list)):
                    x['obs_time'] = x['obs_time'].replace(time_list[24+i], time_list[i])
            for col in col_list :
                for time in time_list :
                    value = day[day['obs_time']==time][col].iloc[0]
                    x[col+time] = value
            nx = x.iloc[:1,15:]
            ny = y.iloc[i:i+1].reset_index(drop=True)
            xy = pd.merge(nx,ny,left_index=True, right_index=True)
            train = pd.concat([train,xy]).reset_index(drop=True)
    return train

df = pd.read_csv("./data/train_input/CASE_01.csv")
def time_value(df):
    ''' 
    ex) 00:59:59 => 01:00:00으로 변환 후 시간단위만 추출
    '''
    df['obs_time'] = pd.to_datetime(df["obs_time"]) + datetime.timedelta(seconds=1)
    df['obs_time'] = df['obs_time'].dt.hour
    return df

df = time_value(df)
df_new = pd.DataFrame()
for i in range(28):
    day = df.iloc[24*i:24*i+24]
    break
value = day['내부온도관측치']
df_1 = pd.DataFrame(value)
print(df_1)
