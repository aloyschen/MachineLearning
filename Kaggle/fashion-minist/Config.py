"""
Introduction:
------------
    配置文件
Author:gaochen
Date:2018.04.07
"""
import os

# 指定数据集路径
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

data_path = './data'

# 训练集路径
train_data_file = os.path.join(data_path, 'fashion-mnist_train.csv')

# 测试集路径
test_data_file = os.path.join(data_path, 'fashion-mnist_test.csv')

# 结果保存路径
output_path = './output'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 图像大小
img_rows, img_cols = 28, 28
channels = 1

# 传统模型训练参数
model_name_param_dict = {'kNN': (KNeighborsClassifier(),
                                 {'n_neighbors': [5, 25, 55]}),
                         'LR': (LogisticRegression(),
                                {'C': [0.01, 1, 100]}),
                         'SVM': (SVC(kernel='linear'),
                                 {'C': [0.01, 1, 100]}),
                         'DT': (DecisionTreeClassifier(),
                                {'max_depth': [50, 100, 150]}),
                         'AdaBoost': (AdaBoostClassifier(),
                                      {'n_estimators': [100, 150, 200]}),
                         'GBDT': (GradientBoostingClassifier(),
                                  {'learning_rate': [0.01, 1, 100]}),
                         'RF': (RandomForestClassifier(),
                                {'n_estimators': [100, 150, 200]})}

# CNN模型路径
cnn_model_dir = './cnn_model'
if not os.path.exists(cnn_model_dir):
    os.makedirs(cnn_model_dir)
# CNN训练参数
batch_size = 256
epochs = 10
iter_nums = 20000
