import glob

def load_data():
    X_tr_list = sorted(glob.glob('../data/train_input/*.csv')) 
    X_te_list = sorted(glob.glob('../data/test_input/*.csv'))
    y_tr_list = sorted(glob.glob('../data/train_target/*.csv'))
    y_te_list = sorted(glob.glob('../data/test_target/*.csv'))

    '''
    print('train :', len(X_tr_list), len(y_tr_list))
    print('test  :', len(X_te_list), len(y_te_list))
    '''