# !user/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 计算交叉熵误差, y和t都是N * 10矩阵，y的10列代表每种数字的可能性，t的10列为one-hot
def cross_entropy_error(y, t):
    # 先判断N即样本数是不是1
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    N = y.shape[0]
    # y和t维度一样，*表示对应位相乘，然后用不指定axis的np.sum，将相乘后的矩阵所有元素相加
    return -np.sum(t * np.log(y + 1e-7)) / N

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)