from feature.clip_base_dataset import *
from feature.after_dataset import *
from glob import glob


def make_data(x_train_path, y_train_path, x_test_path, y_test_path):
    '''
    base_dataset (concat된 Train, Test data)을 일 평균하여 값 반환
    
    pivot_table을 이용하여 6시간 단위 평균값을 반환
    '''
    train_input_list = sorted(glob(x_train_path))
    train_target_list = sorted(glob(y_train_path))

    test_input_list = sorted(glob(x_test_path))
    test_target_list = sorted(glob(y_test_path))


    train, train2 = make_dataset(train_input_list, train_target_list)
    test, test2 = make_dataset(test_input_list, test_target_list)

    train['obs_time'] = train2['obs_time']
    train['DAT'] = train2['DAT']
    test['obs_time'] = test2['obs_time']
    test['DAT'] = test2['DAT']
    
    train = time_value(train)
    test = time_value(test)

    train = limit_range(train)
    test = limit_range(test)

    train = train.fillna(method='ffill')
    test = test.fillna(method='ffill')

    test['predicted_weight_g'] = 0

    train = col_cumsum(train, "시간당분무량", "일간누적분무량")
    train = col_cumsum(train, "시간당백색광량", "일간누적백색광량")
    train = col_cumsum(train, "시간당적색광량", "일간누적적색광량")
    train = col_cumsum(train, "시간당청색광량", "일간누적청색광량")
    train = col_cumsum(train, "시간당총광량", "일간누적총광량")


    test = col_cumsum_test(test, "시간당분무량", "일간누적분무량")
    test = col_cumsum_test(test, "시간당백색광량", "일간누적백색광량")
    test = col_cumsum_test(test, "시간당적색광량", "일간누적적색광량")
    test = col_cumsum_test(test, "시간당청색광량", "일간누적청색광량")
    test = col_cumsum_test(test, "시간당총광량", "일간누적총광량")

    train_temp, test_temp = diff_temp(train, test)
    train_dark, test_dark = none_light(train, test)
    train_water, test_water = water(train, test)

    train = time_split(train)
    test = time_split(test)

    pivot_train = pivot_data(train)
    pivot_test = pivot_data(test)

    train_cumsum = train[['DAT', 'obs_time', '일간누적분무량',
                        '일간누적백색광량', '일간누적적색광량', '일간누적청색광량', '일간누적총광량', 'Case']]
    
    test_cumsum = test[['DAT', 'obs_time', '일간누적분무량',
                        '일간누적백색광량', '일간누적적색광량', '일간누적청색광량', '일간누적총광량', 'Case']]

    train_cumsum = cumsum_group_max(train_cumsum)
    test_cumsum = cumsum_group_max(test_cumsum)

    train_rep = group_median(train)
    test_rep = group_median(test)

    train = concat_df(train_rep, train_cumsum)
    test = concat_df(test_rep, test_cumsum)

    train['diff_temp'] = train_temp
    test['diff_temp'] = test_temp
    train['dark'] = train_dark
    test['dark'] = test_dark
    train['water'] = train_water
    test['water'] = test_water
    
    return train, test, pivot_train, pivot_test
