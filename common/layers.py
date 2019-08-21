# !user/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from common.functions import *
import warnings
warnings.filterwarnings('error')

class ReLu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        try:
            self.mask = (x <= 0) # 生成布尔数组
        except RuntimeWarning:
            print(x)
        out = x.copy() # 深复制一个副本
        out[self.mask] = 0 # 对应True项元素置0
        return out

    def backward(self, dout):
        # 因为调用反向函数之前一定先调用了正向函数，所以这里直接用保存好的mask就行
        dout[self.mask] = 0
        dx = dout
        return dx

class Affine:
    """
    Affine函数不用于ReLu，只需要输入量x
    此类中还包含了自身的一些属性，即W和b，以及dW和db，因此这四个要写入构造函数中
    dx不是本身的属性，不需要写在构造函数里
    """
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return np.dot(self.x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 最终的loss
        self.y = None # softmax输出
        self.t = None # 标签向量

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        # 这里的损失是这次批量所有样本loss的平均值，即平均交叉熵误差
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss # 这是一个数，即平均交叉熵误差

    # 最后一层，默认dout = 1
    def backward(self, dout = 1):
        N = self.t.shape[0]
        # 除以N的目的也是为了取平均值，因为之前求的交叉熵误差都取平均了，这里反向也得平均
        dx = dout * (self.y - self.t) / N
        return dx # 传回去的dx，是一个矩阵，且一定为N * 10，不然传不回去

class BatchNormalization_For_FNN:

    def __init__(self, gamma, beta, momentum = 0.9, running_mean = None, running_var = None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.running_mean = running_mean
        self.running_var = running_var

        # backward
        self.eps = None
        self.var_puls_eps = None
        self.x_ = None
        self.dgamma = None
        self.dbeta = None
        self.batch_size = None

    def forward(self, x, for_trainning):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.ones(D)
            self.running_var = np.zeros(D)

        self.eps = 1e-7
        sample_mean = np.mean(x, axis = 0)
        sample_var = np.var(x, axis = 0)


        # 更新参数
        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * sample_mean
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * sample_var

        if for_trainning:
            self.x_ = (x - sample_mean) / np.sqrt(sample_var + self.eps)
        else:
            self.x_ = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        self.var_puls_eps = sample_var + self.eps

        out = self.gamma * self.x_ + self.beta
        return out

    def backward(self, dout):
        self.batch_size = dout.shape[0]
        self.dgamma = np.sum(dout * self.x_, axis = 0)
        self.dbeta = np.sum(dout, axis = 0)
        dx_ = dout * self.gamma
        dx = (self.batch_size * dx_ - np.sum(dx_, axis = 0) - self.x_ * np.sum(dx_ * self.x_, axis = 0)) / (self.batch_size * np.sqrt(self.var_puls_eps))
        return dx

# 实现Vanilla版本的Dropout
# 对应框架为Chainer
# dropout_rate为删除神经元的比例
class Dropout_Vanilla_Version:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, x, for_trainning):
        if for_trainning:
            # 构造布尔矩阵，True对应的为保留神经元
            self.mask = np.random.rand(*x.shape) > self.dropout_rate
            x = self.mask * x
            return x
        else:
            # 对于测试集，需要乘以一个保留概率以使每层神经元输出的总期望相等
            return x * (1 - self.dropout_rate)

    def backward(self, dout):
        dout = dout * self.mask
        return dout

# 实现Inverted版本的Dropout
# 对应框架为当今主流框架
# 这个版本更好，因为测试时不需要改动代码
class Dropout_Inverted_Version:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, x, for_trainning):
        if for_trainning:
            # 构造布尔矩阵，True对应的为保留神经元
            self.mask = np.random.rand(*x.shape) > self.dropout_rate
            x = self.mask * x
            # 非0元素全部除以1 - p，以保持每层输出期望一致，届时对于测试集无需改动
            return x / (1 - self.dropout_rate)
        else:
            # 对于测试集，无需改动
            return x

    def backward(self, dout):
        dout = dout * self.mask
        return dout