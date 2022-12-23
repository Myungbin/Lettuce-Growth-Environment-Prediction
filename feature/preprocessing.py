import tqdm
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from scipy.signal import butter, lfilter
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob


def preprocessing_2(input_path):
    all_input_list = sorted(glob.glob(input_path))
    for datapath in all_input_list:
        data = pd.read_csv(datapath)  # 이게 지금 28일치 * 24시
        data['obs_time'] = data.index % 24  # 시간통일
        col_list = data.columns
        for i in range(0, len(col_list)):
            col = col_list[i]
            if '누적' in col:
                data[col] = data.groupby((data.obs_time == 0).cumsum()).agg(
                    col_list[i-1]).cumsum()
        data.to_csv(datapath, index=False)


def make_dataset_2(all_input_list_path, all_target_list_path):
    all_input_list = sorted(glob.glob(all_input_list_path))
    all_target_list = sorted(glob.glob(all_target_list_path))

    df_all = pd.DataFrame()
    length = len(all_input_list)
    for idx in range(length):
        X = pd.read_csv(all_input_list[idx])
        y = pd.read_csv(all_target_list[idx])
        y['DAT'] = y['DAT']-1
        df_concat = pd.merge(X, y, on='DAT', how='left')
        df_concat['Case'] = idx+1
        df_all = pd.concat([df_all, df_concat])

    pd.options.display.float_format = '{: .100f}'.format

    df_all['내부온도관측치'][df_all['내부온도관측치'] == 0] = df_all['내부온도관측치'].mean()
    df_all['내부습도관측치'][df_all['내부습도관측치'] == 0] = df_all['내부습도관측치'].mean()

    return df_all


def make_cumsum_columns(df):
    time_list = ['05시', '19시', '23시']
    col_list = ['내부온도관측치누적', '내부습도관측치누적', 'co2관측치누적',
                'ec관측치누적', '분무량누적', '백색광누적', '적색광누적', '청색광누적', '총광량누적']
    for col in col_list:
        for time in time_list:
            df[time+col] = 0


def drop_day_cumsum(df):
    return df.drop(df.filter(regex='일간누적').columns, axis=1)


def make_time_slot(df):
    df['시간대'] = 0
    df['시간대'][(df['obs_time'] >= 0) & (df['obs_time'] <= 5)] = 1
    df['시간대'][(df['obs_time'] > 5) & (df['obs_time'] < 20)] = 2
    df['시간대'][(df['obs_time'] >= 20) & (df['obs_time'] <= 23)] = 3


def filtering(df):

    f1 = df[df['co2관측치'] > 578]['Case'].unique().tolist()
    f2 = df[((df['내부습도관측치'] < 74.4) & (df['내부습도관측치'] > 73.6)) | ((df['내부습도관측치'] < 30) & (
        df['내부습도관측치'] > 27)) | ((df['내부습도관측치'] < 54.5) & (df['내부습도관측치'] > 54.4))]['Case'].unique().tolist()
    f3 = df[(df['일간누적분무량'] < 150)]['Case'].unique().tolist()

    d1 = df[(df['시간당분무량'] > 2100)]['Case'].unique().tolist()
    d2 = df[(df['내부온도관측치'] > 31.124) & (
        df['내부온도관측치'] < 31.126)]['Case'].unique().tolist()
    d3 = df[(df['co2관측치'] < 510.1) & (
        df['co2관측치'] >= 510)]['Case'].unique().tolist()
    d4 = df[(df['시간당백색광량'] > 3094.0) & (
        df['시간당백색광량'] < 3094.2)]['Case'].unique().tolist()
    d5 = df[(df['시간당백색광량'] > 13400) & (df['시간당백색광량']
                                       < 13400.5472)]['Case'].unique().tolist()

    return_arr = list(set(f1) & set(f2) & set(f3))
    for i in set(d1+d2+d3+d4+d5):
        if i in return_arr:
            return_arr.remove(i)

    return return_arr


def weight_moist(df):

    df = df.reset_index()
    case_list = df['Case'].unique()

    df['측정될수분량2'] = 0  # 22시 수분량 + (23시 수분량) * ((22시 습도 + 23시 습도)/2)
    df['측정될수분량1'] = 0  # 23시 수분량 * 23시 습도
    df['측정될수분량3'] = 0  # (22시 수분량 + 23시 수분량) * ((22시 습도 + 23시 습도)/2)

    for i in range(22, len(df), 24):

        s2 = df.loc[i, '시간당분무량'] + df.loc[i+1, '시간당분무량'] * \
            ((df.loc[i, '내부습도관측치'] + df.loc[i+1, '내부습도관측치'])/2)
        s1 = df.loc[i+1, '시간당분무량'] * df.loc[i+1, '내부습도관측치']
        s3 = (df.loc[i, '시간당분무량'] + df.loc[i+1, '시간당분무량']) * \
            ((df.loc[i, '내부습도관측치'] + df.loc[i+1, '내부습도관측치'])/2)
        df.loc[i, '측정될수분량2'] = s2
        df.loc[i, '측정될수분량1'] = s1
        df.loc[i, '측정될수분량3'] = s3

    zerolist = filtering(df)
    for c in case_list:
        if c in zerolist:
            continue
        df.loc[(df['Case'] == c), '측정될수분량2'] = 0
        df.loc[(df['Case'] == c), '측정될수분량1'] = 0
        df.loc[(df['Case'] == c), '측정될수분량3'] = 0

    return df.set_index(keys=['index'], inplace=False, drop=True)


def train_ver(train):
    train_x = train.drop(['predicted_weight_g'], axis=1)
    train_y = train['predicted_weight_g']
    return train_x, train_y


def test_ver(df):
    try:
        df = df.drop(['predicted_weight_g'], axis=1)
    except:
        pass
    return df


def expanding_timeslot(df):
    df = df.groupby(['DAT', 'Case', '시간대']).sum().reset_index()
    df = df.sort_values(by=['Case', 'DAT', '시간대'], axis=0).reset_index()
    df.drop(['index'], axis=1, inplace=True)

    col_list = ['내부온도관측치', '내부습도관측치', 'co2관측치', 'ec관측치', '시간당분무량', '시간당백색광량',
                '시간당적색광량', '시간당청색광량', '시간당총광량']

    pd.options.display.float_format = '{: .44f}'.format

    for col in col_list:
        x = df.groupby(['Case'])[col].expanding(
        ).sum().reset_index().drop(['Case'], axis=1)
        x.drop(['level_1'], axis=1, inplace=True)
        df[col] = x

    return df


def expanding_data(train_x, train_x_2):

    col_list_1 = ['05시내부온도관측치누적', '19시내부온도관측치누적', '23시내부온도관측치누적', '05시내부습도관측치누적',
                  '19시내부습도관측치누적', '23시내부습도관측치누적', '05시co2관측치누적', '19시co2관측치누적',
                  '23시co2관측치누적', '05시ec관측치누적', '19시ec관측치누적', '23시ec관측치누적', '05시분무량누적',
                  '19시분무량누적', '23시분무량누적', '05시백색광누적', '19시백색광누적', '23시백색광누적', '05시적색광누적',
                  '19시적색광누적', '23시적색광누적', '05시청색광누적', '19시청색광누적', '23시청색광누적', '05시총광량누적',
                  '19시총광량누적', '23시총광량누적']

    col_list_2 = ['내부온도관측치', '내부습도관측치', 'co2관측치', 'ec관측치', '시간당분무량', '시간당백색광량',
                  '시간당적색광량', '시간당청색광량', '시간당총광량']

    train_x = train_x.groupby(['DAT', 'Case']).sum().reset_index()  # 784 로
    train_x = train_x.sort_values(by=['Case', 'DAT'], axis=0).reset_index()
    train_x.drop(['index'], axis=1, inplace=True)

    pd.options.display.float_format = '{: .44f}'.format

    k = 0
    for col_2 in col_list_2:
        for i in range(0, 3):
            train_x[col_list_1[k+i]] = train_x_2.loc[train_x_2.index %
                                                     3 == i, col_2].values
        k += 3

    train_x.drop(['obs_time', '시간대', '내부온도관측치', '내부습도관측치',
                 'co2관측치', 'ec관측치'], axis=1, inplace=True)
    train_x = train_x.drop(train_x.filter(regex='시간당').columns, axis=1)

    return train_x


def ec_spray(df):
    pd.options.display.float_format = '{: .100f}'.format

    df['ec_x_분무05'] = (df['05시ec관측치누적']+1) * (df['05시분무량누적']+1)
    df['ec_x_분무19'] = (df['19시ec관측치누적']+1) * (df['19시분무량누적']+1)
    df['ec_x_분무23'] = (df['23시ec관측치누적']+1) * (df['23시분무량누적']+1)
    df['ec_x_분무평균'] = (df['하루평균ec']+1) * (df['하루평균분무량']+1)

    df['적색_+_청색05'] = (df['05시적색광누적']) + (df['05시청색광누적'])
    df['적색_+_청색19'] = (df['19시적색광누적']) + (df['19시청색광누적'])
    df['적색_+_청색23'] = (df['23시적색광누적']) + (df['23시청색광누적'])
    df['적색_+_청색평균'] = (df['하루평균적색광']) + (df['하루평균청색광'])

    return df


def day_mean_value(df):

    df['하루평균온도'] = (df['05시내부온도관측치누적'] + df['19시내부온도관측치누적'] +
                    df['23시내부온도관측치누적']) / 3
    df['하루평균습도'] = (df['05시내부습도관측치누적'] + df['19시내부습도관측치누적'] +
                    df['23시내부습도관측치누적']) / 3
    df['하루평균co2'] = (df['05시co2관측치누적'] + df['19시co2관측치누적'] +
                     df['23시co2관측치누적']) / 3
    df['하루평균ec'] = (df['05시ec관측치누적'] + df['19시ec관측치누적'] + df['23시ec관측치누적']) / 3
    df['하루평균분무량'] = (df['05시분무량누적'] + df['19시분무량누적'] + df['23시분무량누적']) / 3
    df['하루평균백색광'] = (df['05시백색광누적'] + df['19시백색광누적'] + df['23시백색광누적']) / 3
    df['하루평균적색광'] = (df['05시적색광누적'] + df['19시적색광누적'] + df['23시적색광누적']) / 3
    df['하루평균청색광'] = (df['05시청색광누적'] + df['19시청색광누적'] + df['23시청색광누적']) / 3
    df['하루평균총광량'] = (df['05시총광량누적'] + df['19시총광량누적'] + df['23시총광량누적']) / 3

    return df


def weight_moist_sum(df):

    df['수분량합'] = df['측정될수분량1'] + df['측정될수분량2'] + df['측정될수분량3']
    df['수분량합12'] = df['측정될수분량1'] + df['측정될수분량2']
    df['수분량합13'] = df['측정될수분량1'] + df['측정될수분량3']
    df['수분량합23'] = df['측정될수분량2'] + df['측정될수분량3']
    df.drop(['측정될수분량2', '측정될수분량1', '측정될수분량3'], axis=1, inplace=True)

    return df


def kalman_filter(data):

    data = data.drop(data.filter(regex='총광').columns, axis=1)
    data = data.drop(data.filter(regex='백색').columns, axis=1)
    data = data.drop(data.filter(regex='청색').columns, axis=1)
    col_list = data.columns
    case_list = data['Case'].unique()

    for i in range(2, len(col_list)):
        data['kf_X_'+str(i)] = 0

    for i in case_list:
        kal = data[data['Case'] == i]
        for j in tqdm(range(len(col_list))):
            if ((j == 0) | (j == 1)):  # dat, case 뺌
                continue
            sum_c = []
            z = kal.loc[:, kal.columns[j]]
            a = []  # 필터링 된 피쳐(after)
            b = []  # 필터링 전 피쳐(before)
            my_filter = KalmanFilter(dim_x=2, dim_z=1)  # create kalman filter
            # initial state (location and velocity)
            my_filter.x = np.array([[2.], [0.]])
            # state transition matrix
            my_filter.F = np.array([[1., 1.], [0., 1.]])
            my_filter.H = np.array([[1., 0.]])    # Measurement function
            my_filter.P *= 1000.                 # covariance matrix
            my_filter.R = 5                      # state uncertainty
            my_filter.Q = Q_discrete_white_noise(
                dim=2, dt=.1, var=.1)  # process uncertainty
            for k in z.values:
                my_filter.predict()
                my_filter.update(k)
                x = my_filter.x
                a.extend(x[0])
                b.append(k)
            sum_c = sum_c+a
            data['kf_X_'+str(j)][data['Case'] == i] = sum_c
    return data


# 이동 평균, 이동 중간값 구하는 함수
# case 별로 구할 수 있도록 수정

def make_move_mean_median(df, set_amount):

    return_df = pd.DataFrame()

    case_list = df['Case'].unique()
    col_list = df.columns
    for c in case_list:
        target = df[df['Case'] == c]
        for col in col_list:
            mean_arr = []
            median_arr = []
            column_list = target[col].to_list()
            for i in range(set_amount):
                try:
                    mean_arr.append(column_list[i])
                    median_arr.append(column_list[i])
                except:
                    break
            for i in range(set_amount, len(column_list)):
                try:
                    mean_arr.append(
                        float(np.mean(column_list[i-set_amount:i])))
                    median_arr.append(
                        float(np.median(column_list[i-set_amount:i])))
                except:
                    break
            target[f'{col}_mean_{set_amount}'] = mean_arr
            target[f'{col}_median_{set_amount}'] = median_arr
        return_df = pd.concat([return_df, target], axis=0)
    return return_df


def make_move_mean_median_run(df, set1, set2):

    dfc = df.drop(df.filter(regex='kf').columns, axis=1)
    raw_cols = dfc.columns

    df1 = make_move_mean_median(dfc, set1)
    df2 = make_move_mean_median(dfc, set2)

    df1 = df1.drop(raw_cols, axis=1)
    df2 = df2.drop(raw_cols, axis=1)

    df = pd.concat([df, df1], axis=1)
    df = pd.concat([df, df2], axis=1)

    return df


def LPF(df, low, order=1):
    new_df = pd.DataFrame()
    df_x_fill = df.iloc[:, 1:34]
    case_list = df['Case'].unique()
    b, a = butter(
        N=order,
        Wn=low,
        btype='low',
    )
    for c in case_list:
        target = df_x_fill[df_x_fill['Case'] == c]
        lpf_series = lfilter(b, a, target)
        lpf_dataframe = pd.DataFrame(lpf_series)
        new_df = pd.concat([new_df, lpf_dataframe], axis=0)
    new_df = new_df.add_suffix('_LPF')
    new_df = new_df.reset_index(drop=True)
    df = pd.concat([df, new_df], axis=1)

    df.drop(['0_LPF'], axis=1, inplace=True)
    return df
