# -*-coding:utf-8-*-
# Author: gaochen3
# Date: 2018.05.28

import implicit
import pandas as pd
from scipy import sparse
from sklearn import preprocessing

def load_data(path):
    """
    Introduction
    ------------
        读取数据
    Parameters
    ----------
        path: 文件路径
    """
    data = pd.read_csv('../data/push_data_500000', sep='\t', header=None)
    data.columns = ['uid', 'zuid', 'open_num', 'gift', 'like_num', 'comment_num', 'share_num', 'view_time', 'watch_time']
    print('uid num: ', len(set(data['uid'].values)))
    print('zuid num: ', len(set(data['zuid'].values)))


def data_processing(data):
    """
    Introduction
    ------------
        1、对uid, zuid使用label_encoder编码获取其对应的index
        2、对数据进行归一化处理
        3、然后乘以对应权值，将特征映射成0-100内的分数
    Parameters
    ----------
        data: 需要处理的数据
    Returns
    -------
        data: 处理之后的数据
        uid_label_encoder； uid编码，用于后续将index转换为对应的uid
        zuid_label_encoder: zuid编码，用于后续将index转换为对应的zuid
    """
    min_max_Scaler = preprocessing.MinMaxScaler()
    uid_label_encoder = preprocessing.LabelEncoder()
    zuid_label_encoder = preprocessing.LabelEncoder()
    data['uid_index'] = uid_label_encoder.fit_transform(data['uid'].astype(str)).astype(int)
    data['zuid_index'] = zuid_label_encoder.transform(data['zuid'].astype(str)).astype(int)
    numerical_columns = ['open_num', 'gift', 'like_num', 'comment_num', 'share_num', 'view_time', 'watch_time']
    for column in numerical_columns:
        data[column] = min_max_Scaler.fit_transform(data[[column]].values.astype(float))
    data['rating'] = data['open_num'] * 20 + data['gift'] * 10 + data['like_num'] * 10 + data['comment_num'] * 10 + data['share_num'] * 10 + data['view_time'] * 20 + data['watch_time'] * 20
    data = data[data['rating'] > 0]
    return data, uid_label_encoder, zuid_label_encoder


def train(train_data, factors = 100, iterations = 15, regularization = 0.01, use_gpu = False, num_threads = 0, calculate_training_loss = True):
    """
    Introduction
    ------------
        训练协同过滤ALS模型
    Parameters
    ----------
        train_data: 训练集数据
        factors: 矩阵分解中k的维度大小
        iterations: 迭代次数
        regularization: 正则化
        use_gpu: 是否使用gpu
        num_threads: 训练模型使用的线程数
        calculate_training_loss: 是否打印loss的log
    Returns
    -------
        model: 训练好的模型
    """
    model = implicit.als.AlternatingLeastSquares(factors = factors, regularization = regularization, iterations = iterations, use_gpu = use_gpu, num_threads = num_threads, calculate_training_loss = calculate_training_loss)
    # row为zuid, col为uid
    item_user_data = sparse.coo_matrix((train_data['rating'].values, (train_data['zuid_index'].values, train_data['uid_index'].values)))
    model.fit(item_user_data)
    return model


def predict():
    """
    Introduction
    ------------
        
    :return:
    """
