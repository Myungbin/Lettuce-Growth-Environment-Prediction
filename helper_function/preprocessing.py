import pandas as pd


# =================================================================
# preprocessing for ctgan input
def ctgan_preprocessing(df): # input : case df

    # shape of raw data
    raw_shape = df.shape
       
    # input df
    df['obs_time'] = df.index % 24
        
    # limit data range
    df.loc[(df['내부온도관측치'] > 40), '내부온도관측치'] = 40
    df.loc[(df['내부습도관측치'] > 100), '내부습도관측치'] = 100
    df.loc[(df['co2관측치'] > 1200), 'co2관측치'] = 1200
    df.loc[(df['ec관측치'] > 8), 'ec관측치'] = 8
    df.loc[(df['시간당분무량'] > 3000), '시간당분무량'] = 3000
    df.loc[(df['시간당백색광량'] > 120000), '시간당백색광량'] = 120000
    df.loc[(df['시간당적색광량'] > 120000), '시간당적색광량'] = 120000
    df.loc[(df['시간당청색광량'] > 120000), '시간당청색광량'] = 120000
    df.loc[(df['시간당총광량'] > 120000), '시간당총광량'] = 120000
    df.loc[(df['내부온도관측치'] < 0), '내부온도관측치'] = 0
    df.loc[(df['내부습도관측치'] < 0), '내부습도관측치'] = 0
    df.loc[(df['co2관측치'] < 0), 'co2관측치'] = 0
    df.loc[(df['ec관측치'] < 0), 'ec관측치'] = 0
    df.loc[(df['시간당분무량'] < 0), '시간당분무량'] = 0
    df.loc[(df['시간당백색광량'] < 0), '시간당백색광량'] = 0
    df.loc[(df['시간당적색광량'] < 0), '시간당적색광량'] = 0
    df.loc[(df['시간당청색광량'] < 0), '시간당청색광량'] = 0
    df.loc[(df['시간당총광량'] < 0), '시간당총광량'] = 0    
    df['시간당총광량'] = df['시간당청색광량'] + df['시간당백색광량'] + df['시간당적색광량']
        
    cols = df.columns

    # cumsum cols
    for i in range(0, len(cols)):
        col = cols[i]

        if '누적' in col:
            df.drop([col], axis=1, inplace=True)
        
        elif '총광량' in col:
            df.drop([col], axis=1, inplace=True)
        
    print(f'Done. (ctgan preprocessing {raw_shape} => {df.shape})')

    return df


# preprocessing before save
def save_preprocessing(df): # input : case df

    # shape of raw data
    raw_shape = df.shape
        
    # limit data range
    df.loc[(df['내부온도관측치'] > 40), '내부온도관측치'] = 40
    df.loc[(df['내부습도관측치'] > 100), '내부습도관측치'] = 100
    df.loc[(df['co2관측치'] > 1200), 'co2관측치'] = 1200
    df.loc[(df['ec관측치'] > 8), 'ec관측치'] = 8
    df.loc[(df['시간당분무량'] > 3000), '시간당분무량'] = 3000
    df.loc[(df['시간당백색광량'] > 120000), '시간당백색광량'] = 120000
    df.loc[(df['시간당적색광량'] > 120000), '시간당적색광량'] = 120000
    df.loc[(df['시간당청색광량'] > 120000), '시간당청색광량'] = 120000
    df.loc[(df['내부온도관측치'] < 0), '내부온도관측치'] = 0
    df.loc[(df['내부습도관측치'] < 0), '내부습도관측치'] = 0
    df.loc[(df['co2관측치'] < 0), 'co2관측치'] = 0
    df.loc[(df['ec관측치'] < 0), 'ec관측치'] = 0
    df.loc[(df['시간당분무량'] < 0), '시간당분무량'] = 0
    df.loc[(df['시간당백색광량'] < 0), '시간당백색광량'] = 0
    df.loc[(df['시간당적색광량'] < 0), '시간당적색광량'] = 0
    df.loc[(df['시간당청색광량'] < 0), '시간당청색광량'] = 0
        
    cols = df.columns

    # cumsum cols
    for i in range(0, len(cols)):
        col = cols[i]

        if '누적' in col:
            df.drop([col], axis=1, inplace=True)
        
        elif '총광량' in col:
            df.drop([col], axis=1, inplace=True)
        
    print(f'Done. (save {raw_shape} => {df.shape})')

    return df


# =================================================================
# select day to fix
def select_day():

    return

# preprocessing for pred model input
def pred_preprocessing(df):

    return