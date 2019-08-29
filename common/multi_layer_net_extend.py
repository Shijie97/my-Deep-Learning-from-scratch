# !user/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from common.layers import *
from collections import OrderedDict

class MultiLayerNetExtend:

    def __init__(self,
                 input_size,
                 hidden_size_list,
                 output_size,
                 activation = 'ReLu',
                 weight_init_std = 'ReLu',
                 weight_decay_lambda = 0,
                 use_BatchNormalization = False,
                 use_weight_decay = False):
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.output_size = output_size
        self.activation = activation
        self.weight_init_std = weight_init_std
        self.weight_decay_lambda = weight_decay_lambda
        self.use_BatchNormalization = use_BatchNormalization
        self.use_weight_decay = use_weight_decay
        self.params = {} # 一个字典，装的是权重

        # 初始化权重
        self.__init_weight(weight_init_std)

        # 生成层
        self.layers = OrderedDict() # 初始化有序字典，依次装入每一层的实例对象

        # 构造除了最后一层的所有层
        self.construct_layer_except_last()

        # 最后一层
        idx = len(self.hidden_size_list) + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                  self.params['b' + str(idx)])
        self.lastLayer = SoftmaxWithLoss()

    def construct_layer_except_last(self):
        # 依次遍历除了最后一层的所有层
        # 最后一层需要连接SoftMaxWithLoss，没有激活函数和BN，所以单独算
        # idx从1开始是为了序号能从1开始更直观

        activaion_layer = {'Sigmoid': Sigmoid, 'ReLu': ReLu}  # value存的是函数
        for idx in range(1, len(self.hidden_size_list) + 1):
            # Affine层
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            # 将BN层放在激活函数之前
            if self.use_BatchNormalization:
                self.params['gamma' + str(idx)] = np.ones(self.hidden_size_list[idx - 1])
                self.params['beta' + str(idx)] = np.zeros(self.hidden_size_list[idx - 1])
                self.layers['BatchNormalization' + str(idx)] = BatchNormalization_For_FNN(
                    self.params['gamma' + str(idx)],
                    self.params['beta' + str(idx)])
            # 激活函数层
            self.layers['Activation_Function' + str(idx)] = activaion_layer[self.activation]()

    # 根据字符串weight_init_std，来返回相应的高斯函数
    # 这里只有W需要初始化，b初始化一律为0
    def __get_Gauss_Function(self, weight_init_std, number_of_previous_layer, number_of_later_layer):
        gauss_w = None
        gauss_b = np.zeros(number_of_later_layer)
        if weight_init_std == 'ReLu':
            gauss_w = np.random.randn(number_of_previous_layer, number_of_later_layer) * np.sqrt(2.0 / number_of_previous_layer)
        elif weight_init_std == 'Sigmoid':
            gauss_w = np.random.randn(number_of_previous_layer, number_of_later_layer) * np.sqrt(1.0 / number_of_previous_layer)
        else:
            gauss_w = np.random.randn(number_of_previous_layer, number_of_later_layer) * weight_init_std # 这里指的是传进来的weight_init_std不是字符串而是数字
        return gauss_w, gauss_b

    # 根据字符串weight_init_std，来选择相应的初始权重
    def __init_weight(self, weight_init_std):
        all_layer_size = [self.input_size] + self.hidden_size_list + [self.output_size]

        # 对每一层的Affine层进行权值初始化操作
        for idx in range(1, len(all_layer_size)):
            gauss_w, gauss_b = self.__get_Gauss_Function(weight_init_std, all_layer_size[idx - 1], all_layer_size[idx])
            self.params['W' + str(idx)] = gauss_w
            self.params['b' + str(idx)] = gauss_b

    # 前向传播，返回x
    # 注意，predict函数不包括softmax_error那一层
    def predict(self, x, for_trainning = False):
        for layer_name, layer in self.layers.items():
            if 'BatchNormalization' in layer_name:
                x = layer.forward(x, for_trainning)
            else:
                x = layer.forward(x)

        return x

    # 从头到尾遍历（即forward），并求出loss
    def loss(self, x, t, for_trainning = False, use_weight_decay = False):
        y = self.predict(x, for_trainning)
        weight_decay = 0  # 累加器初始化

        if use_weight_decay:
            for idx in range(1, len(self.hidden_size_list) + 2):
                # print(str(idx))
                weight_decay += (self.weight_decay_lambda * np.sum(self.params['W' + str(idx)] ** 2) / 2.0)

        # 求最后一个softmax_error层仅用于求loss
        return self.lastLayer.forward(y, t) + weight_decay

    # 求准确率的时候，需要从头到尾遍历，但是不需要经过softmax_error那一层
    # 只有这个函数既为测试集服务又为训练集服务，其他的都是为训练集服务
    # 1. 对于训练集，一般已经训练完毕，所以for_trainning应该为Flase
    # 2. 对于测试集，无需训练，因此for_trainning也为Flase
    def accuracy(self, x, t):
        y = self.predict(x, for_trainning = False)
        batch_size = x.shape[0]
        y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis = 1)
        accuracy = np.sum(y == t) / float(batch_size)
        return accuracy

    # 求梯度，分三步
    # 1. 前向传播，即调用loss函数，注意，这里的前向传播是要经过softmax_error这一层的，因为要根据误差求梯度，如果求精度就不需要，因为权重已经固定不变了
    # 2. 反向传播，即逆序调用backward函数
    # 3. 求梯度，利用optimizer去求
    def gradient(self, x, t, use_weight_decay = False):

        # 前向传播，注意求梯度这个函数仅用于训练数据集的时候，测试的时候是不会调用这个函数滴
        self.loss(x, t, for_trainning = True, use_weight_decay = use_weight_decay)

        # 反向传播
        dout = 1
        dout = self.lastLayer.backward(dout)
        # reversed函数返回一个反转的迭代器，注意这里是迭代器而不是list
        layers = reversed(self.layers.values()) # 每层的实例对象逆序放入迭代器
        # 注意，在这个反向传播的过程中，每个层都会有dXX这种成员属性，因此这个过程中每个层的这些属性都会同步更新
        # 我们关心的权重只有这四个
        # 1. Affine层的 W 和 b
        # 2. BN层的 gamma 和 beta
        # 之所以几乎所有层都不把dx作为成员属性，是因为我们根本不关心dx是多少，因为dx只起到传播的作用，并不是真正意义上的权重
        for layer in layers:
            dout = layer.backward(dout)

        # 求梯度
        grads = {}
        # 遍历除了最后一层的所有层，这里最后一层指的是最后一个Affine层
        # 我们说的最后一层不是指的softmax_error而是指最后一个Affine，softmax_error紧连着最后一个Affine
        for idx in range(1, len(self.hidden_size_list) + 1):
            # 反向传播求dW的时候，权值衰减的那部分也要算进去
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.params['W' + str(idx)] / 2.0
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db
            if self.use_BatchNormalization:
                grads['gamma' + str(idx)] = self.layers['BatchNormalization' + str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['BatchNormalization' + str(idx)].dbeta

        # 因为最后一个Affine没有BN，所以单独算
        idx = len(self.hidden_size_list) + 1
        grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.params['W' + str(idx)] / 2.0
        grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads




