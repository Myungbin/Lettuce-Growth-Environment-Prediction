import pandas as pd

def ctgan_preprocessing(df_list):

    for df_path in df_list:

        # return df
        df_fin = pd.DataFrame()
        
        # input df
        df = pd.read_csv(df_path)
        df['obs_time'] = df.index % 24 
        df = abs(df)
        
        # data range
        df.loc[(df['내부온도관측치'] > 40), '내부온도관측치'] = 40
        df.loc[(df['내부습도관측치'] > 100), '내부습도관측치'] = 100
        df.loc[(df['co2관측치'] > 1200), 'co2관측치'] = 1200
        df.loc[(df['ec관측치'] > 8), 'ec관측치'] = 8
        df.loc[(df['시간당분무량'] > 3000), '시간당분무량'] = 3000
        df.loc[(df['시간당백색광량'] > 120000), '시간당백색광량'] = 120000
        df.loc[(df['시간당적색광량'] > 120000), '시간당적색광량'] = 120000
        df.loc[(df['시간당청색광량'] > 120000), '시간당청색광량'] = 120000
        df.loc[(df['시간당총광량'] > 120000), '시간당총광량'] = 120000
        df['시간당총광량'] = df['시간당청색광량'] + df['시간당백색광량'] + df['시간당적색광량']
        
        cols = df.columns

        for i in range(0, len(cols)):
            col = cols[i]

            if '누적' in col:
                df[col] = df.groupby((df.obs_time == 0).cumsum()).agg(cols[i - 1]).cumsum()

            df_fin = pd.concat([df_fin, df])

    return df_fin

def pred_preprocessing(df):

    return