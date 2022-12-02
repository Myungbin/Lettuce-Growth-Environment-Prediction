import pandas as pd
from glob import glob
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import glob

def model_train(train, test, model):
    '''
    train shape = (784, ?), test shape = (140, ?)
    model => sklearn api model
    '''

    X = train.drop(['predicted_weight_g', 'Case', 'obs_time', '6time'], axis=1)
    y = train['predicted_weight_g']
    
    print(X.columns)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=113, shuffle=True)
    
    x_test = test.drop(['predicted_weight_g', 'Case', 'obs_time', '6time'], axis=1)
    
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)

    rmse = mean_squared_error(y_val, y_val_pred)**0.5
    print(f"validation rmse: {rmse}")
    
    x_train = train.drop(['predicted_weight_g', 'Case', 'obs_time', '6time'], axis=1)
    y_train = train['predicted_weight_g']
    
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    x_test['predicted_weight_g'] = y_pred
    submit = x_test[['DAT', 'predicted_weight_g']]
    submit['DAT'] = submit['DAT']+1
    all_target_list = sorted(glob.glob('./data/test_target/*.csv'))
    for idx, test_path in enumerate(all_target_list):
        submit_df = pd.read_csv(test_path)
        submit_df['predicted_weight_g'] = submit['predicted_weight_g'][idx*28:idx*28+28].values
        submit_df.to_csv(test_path, index=False)