import pandas as pd
from feature.base_dataset import limit_range
from glob import glob


def moving_average(input_path):
    """make moving_average dataset

    Args:
        input_path (str): data input path

    Returns:
        DataFrame: moving_average time dataset
    """
    input_list = sorted(glob(input_path))
    df_moving = pd.DataFrame()
    for i in input_list:
        df = pd.read_csv(i)
        df = limit_range(df)
        df = df.fillna(method='ffill')
        df = df.drop(['DAT', 'obs_time', '시간당분무량', '시간당백색광량',
                      '시간당적색광량', '시간당청색광량', '시간당총광량',
                      '일간누적청색광량', '일간누적적색광량', '일간누적백색광량'], axis=1)

        ma = df.rolling(6, min_periods=1).mean()
        ma['time'] = [i % 24 for i in range(len(ma))]
        ma['DAT'] = [i//24 for i in range(len(ma))]

        df = pd.pivot_table(ma, index=['DAT'], columns=['time'], aggfunc='mean')
        df.columns = [''.join(str(col)) for col in df.columns]
        df = df.reset_index()

        df_moving = pd.concat([df_moving, df])
    df_moving = df_moving.drop(['DAT'], axis=1)
    return df_moving
