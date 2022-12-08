import pandas as pd
import numpy as np
import datetime


def make_dataset(all_input_list, all_target_list):
    """Train, Test데이터를 하나의 데이터 프레임으로 변경

    Args:
        all_input_list (list): input path list
        all_target_list (list): target path list

    Returns:
        DataFrame: Case01 ~ 28 이 concat된 Dataset
    """
    df_all = pd.DataFrame()
    length = len(all_input_list)
    for idx in range(length):
        X = pd.read_csv(all_input_list[idx])
        y = pd.read_csv(all_target_list[idx])
        y['DAT'] = y['DAT']-1
        df_concat = pd.merge(X, y, on='DAT', how='left')
        df_concat['Case'] = idx+1
        df_all = pd.concat([df_all, df_concat])
    return df_all


def time_value(df):
    """obs_time value 통일

    Args:
        df (DataFrame): train, test data

    Returns:
        DataFrame: obs_time이 통일된 DataFrame
        ex) 00:59:59 => 01:00:00으로 변환 후 시간단위만 추출
    """
    df['obs_time'] = pd.to_datetime(
        df["obs_time"]) + datetime.timedelta(seconds=1)
    df['obs_time'] = df['obs_time'].dt.hour
    return df


def limit_range(df):
    """환경 변수 별 제한 범위를 넘어서는 값을 결측치 처리

    Args:
        df (DataFrame): train, test data

    Returns:
        DataFrame: 제한범위를 벗어난 값을 결측치 처리한 DataFrame
    """
    df.loc[(df['내부온도관측치'] < 4) | (df['내부온도관측치'] > 40), '내부온도관측치'] = np.nan
    df.loc[(df['내부습도관측치'] < 0) | (df['내부습도관측치'] > 100), '내부습도관측치'] = np.nan
    df.loc[(df['co2관측치'] < 0) | (df['co2관측치'] > 1200), 'co2관측치'] = np.nan
    df.loc[(df['ec관측치'] < 0) | (df['ec관측치'] > 8), 'ec관측치'] = np.nan
    df.loc[(df['시간당분무량'] < 0) | (df['시간당분무량'] > 3000), '시간당분무량'] = np.nan
    df.loc[(df['일간누적분무량'] < 0) | (df['일간누적분무량'] > 72000), '일간누적분무량'] = np.nan
    df.loc[(df['시간당백색광량'] < 0) | (df['시간당백색광량'] > 120000), '시간당백색광량'] = np.nan
    df.loc[(df['일간누적백색광량'] < 0) | (
        df['일간누적백색광량'] > 2880000), '일간누적백색광량'] = np.nan
    df.loc[(df['시간당적색광량'] < 0) | (df['시간당적색광량'] > 120000), '시간당적색광량'] = np.nan
    df.loc[(df['일간누적적색광량'] < 0) | (
        df['일간누적적색광량'] > 2880000), '일간누적적색광량'] = np.nan
    df.loc[(df['시간당청색광량'] < 0) | (df['시간당청색광량'] > 120000), '시간당청색광량'] = np.nan
    df.loc[(df['일간누적청색광량'] < 0) | (
        df['일간누적청색광량'] > 2880000), '일간누적청색광량'] = np.nan
    df.loc[(df['시간당총광량'] < 0) | (df['시간당총광량'] > 120000), '시간당총광량'] = np.nan
    df.loc[(df['일간누적총광량'] < 0) | (df['일간누적총광량'] > 2880000), '일간누적총광량'] = np.nan
    return df


def col_cumsum(df, col, cum_col):
    """시간값에 이상치가 있어서 누적값을 새로 생성(train)

    Args:
        df (DataFrame): train data 
        col (str): 시간별 *
        cum_col (str): 일간누적 *

    Returns:
        DataFrame: 누적값이 새로 생성된 DataFrame
    """
    import itertools
    df[cum_col] = 0
    for i in range(784):
        result = itertools.accumulate(df[col][i*24:(i+1)*24])
        cumsum = [value for value in result]
        df[cum_col][i*24:(i+1)*24] = cumsum

    return df


def col_cumsum_test(df, col, cum_col):
    """시간값에 이상치가 있어서 누적값을 새로 생성(test)

    Args:
        df (DataFrame): test data 
        col (str): 시간별 *
        cum_col (str): 일간누적 *

    Returns:
        DataFrame: 누적값이 새로 생성된 DataFrame
    """
    import itertools
    df[cum_col] = 0
    for i in range(140):
        result = itertools.accumulate(df[col][i*24:(i+1)*24])
        cumsum = [value for value in result]
        df[cum_col][i*24:(i+1)*24] = cumsum

    return df


def time_split(df):
    """6시간 단위 시간 분할

    Args:
        df (DataFrame): train, test data

    Returns:
        DataFrame: df['6time']
    """
    df.loc[(df['obs_time'] < 7), '6time'] = '새벽'
    df.loc[(df['obs_time'] >= 7) & (df['obs_time'] < 12), '6time'] = '아침'
    df.loc[(df['obs_time'] >= 12) & (df['obs_time'] < 19), '6time'] = '오후'
    df.loc[(df['obs_time'] >= 19) & (df['obs_time'] <= 24), '6time'] = '저녁'

    return df


def pivot_data(df):
    """6시간 단위의 pivot table(time_split이 선행되어야 함)

    Args:
        df (DataFrame): train, test data

    Returns:
        DataFrame: pivot table data
    """
    df = df.drop(['predicted_weight_g', 'obs_time', '시간당분무량', '시간당백색광량',
                  '시간당적색광량', '시간당청색광량', '시간당총광량', '일간누적총광량'], axis=1)
    df = pd.pivot_table(df, index=['DAT', 'Case'], columns=[
                        '6time'], aggfunc='sum')
    df.columns = [''.join(str(col)) for col in df.columns]
    df = df.reset_index()
    df = df.drop(['DAT', 'Case'], axis=1)
    return df


# 식물이 일정이상크면 같은 물의양이 같은의미를 지니지 않는다.->
