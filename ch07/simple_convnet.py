# !user/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import pickle
from common.layers import *
from collections import OrderedDict

class SimpleConvnet:

    """
    Conv1 -> Pooling1 -> ReLu1 -> Affine1 -> ReLu2 -> Affine2 -> SoftmaxWithLoss
    W1,b1                          W2,b2               W3,b3
    """

    def __init__(self, input_dim = (1, 28, 28), conv_params = None,
                 hidden_size = 100, output_size = 10, weight_init_std = 0.01):

        """

        :param input_dim: 输入图像的三维数据，结构为(C, H, W)，这里通道数为1
        :param conv_params: 卷积层的一些参数，用字典的形式来表示
        :param hidden_size: 全连接层的节点个数
        :param output_size: 输出节点个数
        :param weight_init_std: 初始权重满足的高斯分布权重
        """

        if conv_params == None:
            self.conv_params = {'filter_num' : 30, 'filter_size' : 5,
                                'stride' : 1, 'pad' : 0}
        else:
            self.conv_params = conv_params

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weight_init_std = weight_init_std

        # 获取卷积层参数
        filter_num = conv_params['filter_num']
        filter_size = conv_params['filter_size']
        stride = conv_params['stride']
        pad = conv_params['pad']
        channel_num = input_dim[0]

        # 为了初始化后面的权重矩阵，得知道其行数和列数，因此必须要知道三个东西，下面的宽高是一样的，因为都是正方形
        # 1. 原始图像大小宽高
        # 2. 卷积层输出大小宽高
        # 3. 池化层输出大小宽高

        # 原始图像大小宽高
        input_size = input_dim[1]

        # 卷积层输出大小宽高
        conv_output_size = int((input_size + 2 * pad - filter_size) // stride) + 1

        # 池化层输出大小宽高，这里规定的池化层的ph和pw为2，步幅为2，因此池化层输出宽高应该正好是卷积层输出宽高的一半
        # 池化层后面连一层Affine，所以还要计算池化层所有元素一共有多少个，因此最终计算的是输出的总元素个数，即FN * (FH / 2) * (FW / 2)
        pooling_output_size = filter_num * (conv_output_size // 2) * (conv_output_size // 2)

        # 生成权重，先生成权重才能生成各个层，注意体会这种递进关系
        self.params = {}
        if self.weight_init_std == 'ReLu':
            self.params['W1'] = 0.01 * np.random.rand(filter_num, channel_num, filter_size, filter_size)
            self.params['W2'] = np.sqrt(2.0 / pooling_output_size) * np.random.rand(pooling_output_size, hidden_size)
            self.params['W3'] = np.sqrt(2.0 / hidden_size) * np.random.rand(hidden_size, output_size)
        else:
            self.params['W1'] = weight_init_std * np.random.rand(filter_num, channel_num, filter_size, filter_size)  # FN, C, FH, FW
            self.params['W2'] = weight_init_std * np.random.rand(pooling_output_size, hidden_size)
            self.params['W3'] = weight_init_std * np.random.rand(hidden_size, output_size)

        self.params['b1'] = np.zeros(filter_num) # (FN,)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['b3'] = np.zeros(output_size)

        # 生成各个层
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], stride = stride, pad = pad)
        self.layers['ReLu1'] = ReLu()
        self.layers['Pooling1'] = Pooling(ph = 2, pw = 2, stride = 2, pad = pad)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['ReLu2'] = ReLu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.lastLayer = SoftmaxWithLoss()


    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        loss = self.lastLayer.forward(y, t)
        return loss

    def accuracy(self, x, t):
        """

        :param x: 输入的四维
        :param t: 二维one-hot
        :return: 精度

        注意输入的是四维x，要先通过predict转化成二维的y
        """
        y = self.predict(x)
        if y.ndim == 2:
            y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis = 1)
        acc = np.sum((y == t)) / y.shape[0]
        return acc

    def gradient(self, x, t):
        self.loss(x, t)
        dout = 1
        dout = self.lastLayer.backward(dout)
        reversed_Layer = reversed(self.layers.values())
        for layer in reversed_Layer:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads

    # 参数序列化
    def save_params(self, file_name = 'params.pkl'):
        params = {}
        # 练好的参数传进临时字典params
        for k, v in self.params.items():
            params[k] = v

        # 将临时字典序列化为pickle对象f
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    # 参数反序列化
    def load_params(self, file_name = 'params.pkl'):
        # 反序列化为字典params
        with open(file_name, 'rb') as f:
            params = pickle.load(f)

        # 将本身的self.params更新
        for k, v in params.items():
            self.params[k] = v

        # 光params更新还不够，layers也要更新
        for id, name in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[name].W, self.layers[name].b = self.params['W' + str(id + 1)], self.params['b' + str(id + 1)]


