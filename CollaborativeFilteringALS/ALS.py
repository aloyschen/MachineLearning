# -*-coding:utf-8-*-
# Author: gaochen3
# Date: 2018.05.28

import implicit
import config
import logging
import time
import pandas as pd
from scipy import sparse
from collections import defaultdict
from sklearn import preprocessing


log = logging.getLogger("implicit")

def load_data(path):
    """
    Introduction
    ------------
        读取数据
    Parameters
    ----------
        path: 文件路径
    """
    start = time.time()
    data = pd.read_csv(path, sep='\t', header=None)
    data.columns = ['uid', 'zuid', 'open_num', 'gift', 'like_num', 'comment_num', 'share_num', 'view_time', 'watch_time']
    end = time.time()
    logging.info("read data file in {}".format(end - start))
    logging.info("total uid num: {}".format(len(data['uid'].values)))
    logging.info("total zuid num: {}".format(data['zuid'].values))
    return data


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
    """
    start = time.time()
    min_max_Scaler = preprocessing.MinMaxScaler()
    uid_label_encoder = preprocessing.LabelEncoder()
    zuid_label_encoder = preprocessing.LabelEncoder()
    data['uid_index'] = uid_label_encoder.fit_transform(data['uid'].astype(str)).astype(int)
    data['zuid_index'] = zuid_label_encoder.fit_transform(data['zuid'].astype(str)).astype(int)
    numerical_columns = ['open_num', 'gift', 'like_num', 'comment_num', 'share_num', 'view_time', 'watch_time']
    for column in numerical_columns:
        data[column] = min_max_Scaler.fit_transform(data[[column]].values.astype(float))
    data['rating'] = data['open_num'] * 20 + data['gift'] * 10 + data['like_num'] * 10 + data['comment_num'] * 10 + data['share_num'] * 10 + data['view_time'] * 20 + data['watch_time'] * 20
    data = data[data['rating'] > 0]
    end = time.time()
    logging.info("data processing in {}".format(end - start))
    return data


def calculate_similar_zuid(data_path, save_path, factors = 100, iterations = 15, regularization = 0.01, use_gpu = False, num_threads = 0, calculate_training_loss = True):
    """
    Introduction
    ------------
        使用协同过滤ALS模型计算相似主播，然后扩充push粉丝库
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

    """
    start = time.time()
    result = defaultdict(list)
    zuid_index = {}
    similar_zuids = {}
    data = load_data(data_path)
    data = data_processing(data)
    model = implicit.als.AlternatingLeastSquares(factors = factors, regularization = regularization, iterations = iterations, use_gpu = use_gpu, num_threads = num_threads, calculate_training_loss = calculate_training_loss)
    # row为zuid, col为uid
    item_user_data = sparse.coo_matrix((data['rating'].values, (data['zuid_index'].values, data['uid_index'].values)))
    model.fit(item_user_data)
    uid_zuid_df = data[['uid', 'zuid', 'zuid_index']]
    for index, row in uid_zuid_df.iterrows():
        zuid_index[row['zuid_index']] = row['zuid']
        similar_zuid_index = model.similar_items(row['zuid_index'])
        similar_zuids[row['zuid']] = [element[0] for element in similar_zuid_index]
    uid_zuid_df = uid_zuid_df.groupby(['uid']).apply(lambda tdf: pd.Series(dict([[column, tdf[column].unique().tolist()] for column in tdf])))
    for index, row in uid_zuid_df.iterrows():
        for zuid in row['zuid']:
            if zuid in similar_zuids.keys():
                result[row['uid'][0]] = [zuid_index[element] for element in similar_zuids[zuid]]
        result[row['uid'][0]] = list(set(result[row['uid'][0]]))
    end = time.time()
    logging.info("calculate similar zuid in {}".format(end - start))
    save_uid_zuid(result, save_path)
    logging.info("save data successfully")


def save_uid_zuid(data, path):
    """
    Introduction
    ------------
        将计算出来的主播uid和粉丝uid写入文件中
    Parameters
    ----------
        data: 存储用户uid和主播uid关系对的数据
        path: 保存文件路径
    Returns
    -------
        None
    """
    with open(path, 'w+', encoding = 'utf-8') as file:
        for uid in data.keys():
            for zuid in data[uid]:
                file.write(str(uid) + '\t' + str(zuid))


if __name__ == '__main__':
    calculate_similar_zuid(config.data_file, config.save_file, config.factors, config.iterations, config.regularization, config.use_gpu, config.num_threads, config.calculate_training_loss)












