from glob import glob
from feature.preprocessing import *


def make_data(input_path_train, target_path_train, input_path_test, target_path_test):

    pd.options.display.float_format = '{: .20f}'.format

    preprocessing_2(input_path_train)
    preprocessing_2(input_path_test)

    train = make_dataset_2(input_path_train, target_path_train)
    test = make_dataset_2(input_path_test, target_path_test)

    make_cumsum_columns(train)
    make_cumsum_columns(test)

    make_time_slot(train)
    make_time_slot(test)

    train = weight_moist(train)
    test = weight_moist(test)

    train = drop_day_cumsum(train)
    test = drop_day_cumsum(test)

    train_x, train_y = train_ver(train)
    test_x = test_ver(test)

    train_x_ex = expanding_timeslot(train_x)
    test_x_ex = expanding_timeslot(test_x)

    train_x = expanding_data(train_x, train_x_ex)
    test_x = expanding_data(test, test_x_ex)

    train_x = day_mean_value(train_x)
    test_x = day_mean_value(test_x)

    train_x = ec_spray(train_x)
    test_x = ec_spray(test_x)

    train_x = weight_moist_sum(train_x)
    test_x = weight_moist_sum(test_x)

    train_x_kf = kalman_filter(train_x)
    test_x_kf = kalman_filter(test_x)

    train_x = make_move_mean_median_run(train_x_kf, 7, 14)
    test_x = make_move_mean_median_run(test_x_kf, 7, 14)

    train_x = LPF(train_x, 0.1, 1)
    test_x = LPF(test_x, 0.1, 1)

    train_x = train_x.drop(train_x.filter(regex='Case'), axis=1)
    test_x = test_x.drop(test_x.filter(regex='Case'), axis=1)

    return train_x, train_y, test_x
