import glob


# load tr, te file list
def load_data():
    
    X_tr_list = sorted(glob.glob('./data/train_input/*.csv')) 
    X_te_list = sorted(glob.glob('./data/test_input/*.csv'))
    y_tr_list = sorted(glob.glob('./data/train_target/*.csv'))
    y_te_list = sorted(glob.glob('./data/test_target/*.csv'))
    
    '''
    print('train :', len(X_tr_list), len(y_tr_list))
    print('test  :', len(X_te_list), len(y_te_list))
    '''
    
    return X_tr_list, X_te_list, y_tr_list, y_te_list

# load tr, te file list
def load_data_aug():
    
    X_tr_list = sorted(glob.glob('./data/aug_train_input/*.csv')) 
    X_te_list = sorted(glob.glob('./data/aug_test_input/*.csv'))
    
    return X_tr_list, X_te_list

# aug data => add cumsum cols => return X, y
def make_data():

    return