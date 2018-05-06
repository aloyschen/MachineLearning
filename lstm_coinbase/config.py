# 训练数据路径
data_file = './data/price.csv'
# 数据的统计值列
stats_cols = ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)']
# 原始数据的标签列
raw_label_col = 'Weighted_Price'
# 开始预测年份
year_start_pred = 2017
# batch大小
batch_size = 1
# 隐藏层单元数量
hidden_num = 100
# 时间序列长度
time_step = 1
# 训练迭代次数
nb_epoch = 10
# 模型保存路径
model_file = './model/train.h5'