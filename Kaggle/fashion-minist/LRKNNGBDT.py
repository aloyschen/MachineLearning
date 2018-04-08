"""
Introduction:
------------
    Kaggle图像分类数据集fashion_mnist测试
    使用传统的分类器对图像进行分类
    将28*28的图片转成一行特征向量进行处理
Author: gaochen
Date: 2018.04.06
"""
import cv2
import Config
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

def load_data(dataPath):
    """
    读取训练集数据或者测试集数据
    Parameters
    ----------
        dataPath: 文件路径
    Returns
    -------
        数据集特征和标签
    """
    data = pd.read_csv(dataPath)
    X = data.iloc[:, 1:].values.astype(np.uint8)
    y = data.iloc[:, 0].values.astype(np.uint8)
    return X, y


def feature(data_X):
    """
    对数据集样本进行处理
    Parameters
    ----------
        data_X: 数据集样本，shape为[n_samples, rows*cols]
    """
    n_samples = data_X.shape[0]
    feature = []
    for i in range(n_samples):
        image = data_X[i, :].reshape(Config.img_rows, Config.img_cols)
        image = cv2.medianBlur(image, 3)
        image = cv2.equalizeHist(image)
        feature.append(image.flatten())
        if i % 10000 == 0:
            print("提取特征进度：{}/{}".format(i, data_X.shape[0]))
    return np.array(feature)


def train():
    """
    针对训练集数据进行训练
    """
    print("读取训练集数据和测试集数据")
    train_X, train_y = load_data(Config.train_data_file)
    test_X, test_y = load_data(Config.test_data_file)
    print("提取训练集特征")
    train_X = feature(train_X)
    print("提取测试集特征")
    test_X = feature(test_X)
    for model_name, (model, param) in Config.model_name_param_dict.items():
        clf = GridSearchCV(model, param, scoring = "accuracy", cv = 3, refit = True)
        start = time.time()
        print("开始训练模型：{}".format(model_name))
        clf.fit(train_X, train_y)
        # 计时
        end = time.time()
        duration = end - start
        # 验证模型
        print('训练准确率：{:.3f}'.format(clf.score(train_X, train_y)))
        score = clf.score(test_X, test_y)
        print('测试准确率：{:.3f}'.format(score))
        print('训练模型耗时≥; '
              ': {:.4f}s'.format(duration))

if __name__ == "__main__":
    train()
