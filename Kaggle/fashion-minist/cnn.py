"""
Introduction
------------
    该脚本使用tensorflow
    构建CNN模型进行图片分类
Author: gaochen
Date: 2018.04.08
"""
import os
import numpy as np
import pandas as pd
import Config
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = str(Config.GPU)
def load_data(path):
    """
    读取数据集数据
    Parameters
    ----------
        path: 文件路径
    Returns
    -------
        X: 数据集特征
        y: 数据集样本
    """
    features, labels = [], []
    data = pd.read_csv(path)
    sample_numbers = data.shape[0]
    for num in range(sample_numbers):
        feature = data.iloc[num, :-1].values.astype(np.float32).reshape(Config.img_rows, Config.img_cols)
        label = data.iloc[num, 0].astype(np.int32)
        features.append(feature)
        labels.append(label)
    return np.asarray(features), np.asarray(labels)


def cnn_model(features, labels, mode):
    """
    构建CNN模型:
        -conv1
        -pool1
        -conv2
        -pool2
        -dense
        -dropout
        -logits
    """
    print(features)
    features = tf.reshape(features['x'], [-1, Config.img_rows, Config.img_cols, Config.channels])

    # 卷积层 #1
    conv1 = tf.layers.conv2d(inputs = features, filters = 32, kernel_size = [5, 5], kernel_initializer = tf.truncated_normal_initializer(stddev = 0.1), padding = 'same', activation = tf.nn.relu)

    # 池化层 #1
    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2, 2], strides = 2)

    # 卷积层 #2
    conv2 = tf.layers.conv2d(inputs = pool1, filters = 64, kernel_size = [5, 5], kernel_initializer = tf.truncated_normal_initializer(stddev = 0.1), padding = 'same', activation = tf.nn.relu)

    # 池化层 #2
    pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2, 2], strides = 2)

    # 全连接层 #1
    pool2_flatten = tf.reshape(pool2, [-1, 7*7*64])
    dense = tf.layers.dense(inputs = pool2_flatten, units = 1024, activation = tf.nn.relu)

    # dropout
    dropout = tf.layers.dropout(inputs = dense, rate = 0.4, training = mode == tf.estimator.ModeKeys.TRAIN)

    # logits层
    logits = tf.layers.dense(inputs = dropout, units = Config.n_classes)

    predictions = {
        'classes' : tf.argmax(input = logits, axis = 1),
        'probabilitues' : tf.nn.softmax(logits, name = 'softmax_tensor')
    }
    # 如果是预测直接输出预测结果
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)
    # 定义损失函数loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
        train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)
    # 加入评估指标
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def train():
    """
    使用estimator.Estimator构建训练器，训练cnn模型
    """
    train_features, train_labels = load_data(Config.train_data_file)
    eval_features, eval_labels = load_data(Config.test_data_file)
    fishion_classifier = tf.estimator.Estimator(model_fn = cnn_model, model_dir = Config.cnn_model_dir)
    #设置打印log参数
    tf.logging.set_verbosity(tf.logging.INFO)
    #tensors_to_log = {"probabilities" : "softmax_tensor"}
    #logging_hook = tf.train.LoggingTensorHook(tensors = tensors_to_log, every_n_iter = 50)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x" : train_features},
        y = train_labels,
        batch_size = Config.batch_size,
        num_epochs = Config.epochs,
        shuffle = True
    )
    fishion_classifier.train(train_input_fn, steps = Config.iter_nums)
    #评估测试集
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x" : eval_features},
        y = eval_labels,
        num_epochs = 1,
        shuffle = False
    )
    eval_results = fishion_classifier.evaluate(eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    train()
