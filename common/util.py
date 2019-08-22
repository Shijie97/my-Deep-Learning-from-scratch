# !user/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

def shuffle_dataset(x, t):

    permutation = np.random.permutation(x.shape[0])
    x = x[permutation, :]
    t = t[permutation]
    return x, t # 这里返回是因为这个函数不能改变实参，只能改变形参