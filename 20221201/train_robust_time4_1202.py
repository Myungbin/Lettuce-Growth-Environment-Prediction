def make_dataset(input_path, target_path): ## train_robust_time4_1202
    train = pd.DataFrame()
    all_input_list = sorted(glob.glob(input_path))
    all_target_list = sorted(glob.glob(target_path))
    for x, y in zip(all_input_list,all_target_list):
        x = pd.read_csv(x)
        y = pd.read_csv(y)
        x = x.iloc[:,1:]
        x = x.drop(['일간누적분무량합', '일간누적백색광량합', '일간누적적색광량합', '일간누적청색광량합', '일간누적총광량합','일간누적총광량','시간당총광량'],axis=1)
        col_list = x.columns[1:]
        for i in range(0,28) :
            day = x.iloc[24*i:24*i+24]

            day_white = day[day['일간누적백색광량']>=0]['일간누적백색광량'].max()
            day_water = day[day['일간누적분무량']>=0]['일간누적분무량'].max()
            day_blue = day[day['일간누적청색광량']>=0]['일간누적청색광량'].max()
            day_red = day[day['일간누적적색광량']>=0]['일간누적적색광량'].max()

            day_white2 = day[day['일간누적백색광량']>=0]['일간누적백색광량'].mean()
            day_water2 = day[day['일간누적분무량']>=0]['일간누적분무량'].mean()
            day_blue2 = day[day['일간누적청색광량']>=0]['일간누적청색광량'].mean()
            day_red2 = day[day['일간누적적색광량']>=0]['일간누적적색광량'].mean()

            time_list = day['obs_time'].unique()

            for col in col_list :
                if col in ['일간누적백색광량','일간누적청색광량','일간누적적색광량','일간누적분무량','일간누적총광량',
                            '일간누적분무량합', '일간누적백색광량합', '일간누적적색광량합', '일간누적청색광량합', '일간누적총광량합']:
                    for t in range(11,len(time_list)):
                        time = time_list[t]
                        value1 = day[day['obs_time']==time][col].iloc[0]
                        x[col+str(time)] = value1
                else :
                    for t in range(0,len(time_list)-3) :
                        time = time_list[t]
                        ntime = time_list[t+1]
                        nntime = time_list[t+2]
                        nnntime = time_list[t+3]
                        value1 = day[day['obs_time']==time][col].iloc[0]
                        value2 = day[day['obs_time']==ntime][col].iloc[0]
                        value3 = day[day['obs_time']==nntime][col].iloc[0]
                        value4 = day[day['obs_time']==nnntime][col].iloc[0]
                        x[col+str(time)+str("~")+str(nnntime)] = (value1 + value2 + value3 + value4)

            x['day_water_weight_max'] = (day_water*(i+1))
            x['day_white_weight_max'] = (day_white*(i+1))
            x['day_red_weight_max'] = (day_red*(i+1))
            x['day_blue_weight_max'] = (day_blue*(i+1))

            x['day_water_weight_mean'] = (day_water2*(i+1))
            x['day_white_weight_mean'] = (day_white2*(i+1))
            x['day_red_weight_mean'] = (day_red2*(i+1))
            x['day_blue_weight_mean'] = (day_blue2*(i+1))

            nx = x.iloc[:1,15:]
            ny = y.iloc[i:i+1].reset_index(drop=True)
            xy = pd.merge(nx,ny,left_index=True, right_index=True)
            train = pd.concat([train,xy]).reset_index(drop=True)

    return train
