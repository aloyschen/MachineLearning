import config
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
def load_data():
    """
    读取数据，按天重采样
    """
    data = pd.read_csv(config.data_file)
    data['date'] = pd.to_datetime(data['Timestamp'], unit = 's')
    data.set_index('date', inplace = True)
    data.sort_index(inplace = True)
    data = data.resample('1D').mean()
    data.dropna(inplace = True)
    data = data[config.stats_cols + [config.raw_label_col]]
    return data

def preprocessing(data):
    """
    对数据进行预处理, 先做差分，然后用前一天的数据作特征和标签
    """
    data = data.diff()
    data = data.shift(-1)
    data.fillna(0, inplace = True)
    # year_start_pred - 1 前（包括）的时序数据作为训练数据
    train_data = data.loc[:str(config.year_start_pred - 1)]
    test_data = data.loc[str(config.year_start_pred) : ]
    train_X = train_data[config.stats_cols].values
    train_y = train_data[config.raw_label_col].values.reshape(-1, 1)
    test_X = test_data[config.stats_cols].values
    test_y = test_data[config.raw_label_col].values.reshape(-1, 1)
    train_scaler = MinMaxScaler(feature_range = (-1, 1))
    train_X = train_scaler.fit_transform(train_X)
    test_X = train_scaler.transform(test_X)
    lable_scaler = MinMaxScaler(feature_range = (-1, 1))
    train_y = lable_scaler.fit_transform(train_y)
    test_y = lable_scaler.transform(test_y)

    train_num = train_X.shape[0]
    test_num = test_X.shape[0]
    input_dim = train_X.shape[1]
    train_X = train_X.reshape(train_num, config.time_step, input_dim)
    test_X = test_X.reshape(test_num, config.time_step, input_dim)
    return lable_scaler, train_X, train_y, test_X, test_y, test_data
