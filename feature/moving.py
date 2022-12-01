import pandas as pd

def moving_average(input_list):
    
    df_moving = pd.DataFrame()
    for i in input_list:
        df = pd.read_csv(i)
    
        df = df.drop(['DAT', 'obs_time', '일간누적분무량', '일간누적백색광량',
                    '일간누적적색광량', '일간누적청색광량', '일간누적총광량'], axis=1)
        
        ma = df.rolling(6, min_periods=1).mean()
        ma['time'] = [i%24 for i in range(len(ma))]
        ma['DAT'] = [i//24 for i in range(len(ma))]

        df = pd.pivot_table(ma, index=['DAT'], columns=['time'], aggfunc='mean')
        df.columns = [''.join(str(col)) for col in df.columns]
        df = df.reset_index()    
        
        df_moving = pd.concat([df_moving, df])
    return df_moving
    