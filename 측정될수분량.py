def weight_moist(df):
    df = df.reset_index()

    df['측정될수분량2'] = 0 # 22시 수분량 + (23시 수분량) * ((22시 습도 + 23시 습도)/2)
    df['측정될수분량1'] = 0 # 23시 수분량 * 23시 습도
    df['측정될수분량3'] = 0 # (22시 수분량 + 23시 수분량) * ((22시 습도 + 23시 습도)/2)

    for i in range(22,len(df),24) : 

        s2 = df.loc[i,'시간당분무량'] + df.loc[i+1,'시간당분무량'] * ((df.loc[i,'내부습도관측치'] + df.loc[i+1,'내부습도관측치'])/2)
        s1 = df.loc[i+1,'시간당분무량'] * df.loc[i+1,'내부습도관측치'] 
        s3 = (df.loc[i,'시간당분무량'] + df.loc[i+1,'시간당분무량']) * ((df.loc[i,'내부습도관측치'] + df.loc[i+1,'내부습도관측치'])/2)
        df.loc[i,'측정될수분량2'] = s2
        df.loc[i,'측정될수분량1'] = s1
        df.loc[i,'측정될수분량3'] = s3

    return df.set_index(keys=['index'], inplace=False, drop=True)
