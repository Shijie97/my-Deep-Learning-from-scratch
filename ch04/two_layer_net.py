# !user/bin/python
# -*- coding: UTF-8 -*-
import sys, os
sys.path.append(os.pardir)
from common.gradient import numerical_gradient
import numpy as np
from common.functions import *

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    # 预测函数，x为输入样本矩阵，y为输出矩阵，N * 10，N为输入样本数，10为输出空间
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = sigmoid(a2)
        return y

    # 计算损失loss，x为输入，y为监督
    def loss(self, x, t):
        # 先根据输入，前向传播之后的输出矩阵N * 10
        y = self.predict(x)
        return cross_entropy_error(y, t) # 这是一个数，是针对这N个样本的总损失

    # 求准确率，输入x为样本矩阵N * 768， t为监督样本矩阵N * 10，t是one-hot表示的
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1) # axis = 1代表无视列，求每一行最大的那个的索引
        t = np.argmax(t, axis = 1)
        N = y.shape[0]
        return np.sum(y == t) / float(N) # y == t先转化成布尔数组，sum计算True的数量

    # W1， W2， b1, b2根据梯度下降进行更新
    # 输入x为样本矩阵N * 768， t为监督样本矩阵N * 10，t是one-hot表示的
    # 这个函数返回更新后的字典，字典里面包含各个参数矩阵对应项的梯度
    def numerical_gradient_in_class(self, x, t):

        # 定义的这个f函数参数W是伪参，因为f的输入应该是x和t，但是我要求关于W和b的偏导
        # 只能这样做，现将W11 + h，然后求一遍损失
        # 再讲W11 - h，再求一遍损失
        # 最后求上面的差再除以2h，就是对于W11的偏导
        # W是隐变量，不好求导，所以要W不能作为真正的参数带进去
        # 执行f(W)的过程，其实是计算了一遍交叉熵误差的过程，W的改变会影响最后的误差结果
        def f(W):
            return self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(f, self.params['W1'])
        grads['b1'] = numerical_gradient(f, self.params['b1'])
        grads['W2'] = numerical_gradient(f, self.params['W2'])
        grads['b2'] = numerical_gradient(f, self.params['b2'])

        return grads

    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads