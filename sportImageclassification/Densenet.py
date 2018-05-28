import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Densenet():
    def __init__(self):
        """

        """
        self.is_training = tf.placeholder(tf.bool, shape=[])
    def bias_variable(self, shape, name):
        """
        Introduction
        ------------
            偏置初始化
        Parameters
        ----------
            shape: 偏置矩阵大小
            name: 变量名字
        Returns
        -------
            bias: 偏置项
        """
        initial = tf.constant(0.0, shape = shape)
        return tf.get_variable(initializer = initial, name = name)


    def weight_variable_xavier(self, shape, name):
        """
        Introduction
        ------------
            使用xavier方法初始化权重矩阵
        Parameters
        ----------
            shape: 权重矩阵大小
            name: 变量名字
        Returns
        -------
            weight: 权重矩阵
        """
        return tf.get_variable(name = name, shape = shape, initializer = tf.contrib.layers.xavier_initializer)


    def batch_norm_layers(self, x, scope, moving_decay=0.999):
        """
        Introduction
        ------------
            batch norm层
        Parameters
        ----------
            x: 输入变量
            scope: 变量命名范围
            moving_decay: 滑动平均衰减因子
        """
        with tf.variable_scope(scope):
            beta = tf.Variable(tf.constant(0.0, shape = [x.shape[-1]], name = 'beta', trainable = True))
            gamma = tf.Variable(tf.constant(1.0, shape = [x.shape[-1]], name = 'gamma', trainable = True))
            axises = np.arange(len(x.shape) - 1)
            batch_mean, batch_var = tf.nn.moments(x, axises, name = 'moments')
            ema = tf.train.ExponentialMovingAverage(moving_decay)
            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(self.is_training, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed

    def Relu(self, x):
        """
        Introduction
        ------------
            Relu函数
        Parameters
        ----------
            x: 输入变量
        """
        return tf.nn.relu(x)

    def Average_pooling(self, x, pool_size, stride = 2, padding = 'VALID'):
        """
        Introduction
        ------------
            平均值池化层
        Parameters
        ----------
            x: 输入变量
            pool_size: 池化kernel的大小
            stride: 池化层的步长
            padding: 池化层的padding
        Returns
        -------
            平均值池化层
        """
        return tf.nn.avg_pool(x, pool_size, strides = stride, padding = padding)


    def transition_layers(self, input, scope):
        """
        Introduction
        ------------
            每个dense block之间的降采样层, 包括batch norm, 1x1 conv, 2x2 average pool
        Parameters
        ----------
            input: transition层输入
        """



def move_averange():
    with tf.variable_scope('move_averange'):
        x = tf.Variable(0.0, name = 'x')
        x_plus = tf.assign_add(x, 1)
        ema = tf.train.ExponentialMovingAverage(decay=0.1)
        result, orign = [], []
        with tf.control_dependencies([x_plus]):
            ema_apply_op = ema.apply([x])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(5):
            sess.run(ema_apply_op)
            ema_x = sess.run(ema.average(x))
            true_x = sess.run(x)
            result.append(ema_x)
            orign.append(true_x)
            print(true_x, ema_x)
    plt.plot(range(5), result, 'r')
    plt.plot(range(5), orign, 'b')
    plt.show()


M = np.array([
        [[1],[-1],[0]],
        [[-1],[2],[1]],
        [[0],[2],[-2]]
    ])

print ("Matrix shape is: ",M.shape)

filter_weight = tf.get_variable('weights', [2, 2, 1, 1], initializer = tf.constant_initializer([ [1, -1], [0, 2]]))
biases = tf.get_variable('biases', [1], initializer = tf.constant_initializer(1))

M = np.asarray(M, dtype='float32')
M = M.reshape(1, 3, 3, 1)
print("Reshaped M: \n", M)

x = tf.placeholder('float32', [1, None, None, 1])
# 在3x3的M的右边和下面补零，卷积窗口每次移动两个单位。得到的conv为2x2的矩阵，再转成高维矩阵--[1, 2, 2, 1]
conv = tf.nn.conv2d(x, filter_weight, strides = [1, 2, 2, 1], padding = 'SAME')
# 将卷积求得的值加上偏置就为所求值
bias = tf.nn.bias_add(conv, biases)
# 在3x3的M的右边和下面不补零，卷积窗口从左上角每次移动两个单位，遇到数据不足时，只取当前数据。
# 得到的pool为2x2的矩阵，再转成高维矩阵--[1, 2, 2, 1]
pool = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    convoluted_M = sess.run(bias,feed_dict={x:M})
    pooled_M = sess.run(pool,feed_dict={x:M})
    print ("convoluted_M: \n", convoluted_M.reshape(2, 2))
    print ("pooled_M: \n", pooled_M.reshape(2, 2))

