# !user/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags = ['multi_index'], op_flags = ['readwrite'])
    while not it.finished:
        temp_idx = x[it.multi_index]

        # 先算f(x + h)
        # 这里其实还涉及到一个函数传参的问题
        # 这里传进来的x是对象network的params[XXX]，传进来的时候，x和params[XXX]指向同一片内存
        # 如果在这里改变x，则network.params[XXX]也会同样变化
        # 此外，我们的f是loss函数，相当于要求一遍两层网络的loss，这里用到的矩阵，一定是刚刚改变过的
        x[it.multi_index] = float(temp_idx) + h
        fxh1 = f(x[it.multi_index])

        # 再算f(x - h)
        x[it.multi_index] = float(temp_idx) - h
        fxh2 = f(x[it.multi_index])

        # 计算导数
        grad[it.multi_index] = (fxh1 - fxh2) / float(2.0 * h)
        x[it.multi_index] = temp_idx
        it.iternext()
    return grad