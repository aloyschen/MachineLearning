import utils
import config
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model

class LSTM_Model:
    def __init__(self, input_dim, time_step, batch_size, hiddenn_num):
        self.input_dim = input_dim
        self.time_step = time_step
        self.batch_size = batch_size
        self.hidden_num = hiddenn_num
    def build(self):
        model = Sequential()
        model.add(LSTM(self.hidden_num, batch_input_shape = (self.batch_size, self.time_step, self.input_dim), stateful=True))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

def train():
    data = utils.load_data()
    label_scaler, train_X, train_y, test_X, test_y, _ = utils.preprocessing(data)
    input_dim = train_X.shape[2]
    model = LSTM_Model(input_dim, config.time_step, config.batch_size, config.hidden_num).build()
    model.fit(train_X, train_y, batch_size = config.batch_size, epochs = config.nb_epoch, shuffle = False, validation_data = (test_X, test_y))
    mse = model.evaluate(test_X, test_y, batch_size = config.batch_size)
    print('Test mse: {}'.format(mse))
    model.save(config.model_file)


def predict():
    model = load_model(config.model_file)
    # 验证模型
    data = utils.load_data()
    label_scaler, _, _, test_X, test_y, test_data = utils.preprocessing(data)
    test_dates = test_data.index
    pred_daily_df = pd.DataFrame(columns=['True Value', 'Pred Value'], index=test_dates)

    for i, test_date in enumerate(test_dates):
        X = test_X[i].reshape(1, config.time_step, test_X.shape[2])
        y_pred = model.predict(X, batch_size = config.batch_size)[0]
        # scale反向操作，恢复数据范围
        rescaled_y_pred = label_scaler.inverse_transform(y_pred.reshape(-1, 1))[0, 0]

        # 差分反向操作，恢复数据的值：加上前一天的真实标签
        previous_date = test_date - pd.DateOffset(days=1)
        recoverd_y_pred = rescaled_y_pred + data.loc[previous_date][config.raw_label_col]
        true_value = test_data.loc[test_date][config.raw_label_col] + data.loc[previous_date][config.raw_label_col]

        # 保存数据
        pred_daily_df.loc[test_date, 'Pred Value'] = recoverd_y_pred
        pred_daily_df.loc[test_date, 'True Value'] = true_value
        print('Date={}, 真实值={}, 预测值={}'.format(test_date, true_value, recoverd_y_pred))

    pred_daily_df.plot()
    plt.show()
predict()



