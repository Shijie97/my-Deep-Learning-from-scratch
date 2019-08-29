# !user/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

def shuffle_dataset(x, t):

    permutation = np.random.permutation(x.shape[0])
    x = x[permutation, :]
    t = t[permutation]
    return x, t # 这里返回是因为这个函数不能改变实参，只能改变形参


def im2col(input_data, fh, fw, stride = 1, pad = 0):

    """

    :param input_data: 输入数据，四维，分别是(N, C, H, W)
    :param fh: 滤波器高
    :param fw: 滤波器宽
    :param stride: 步幅
    :param pad: 填充行/列数

    :return: 二维矩阵col
    """

    N, C, H, W = input_data.shape

    # 计算输出的out_h和out_w
    out_h = (H + 2 * pad - fh) // stride + 1
    out_w = (W + 2 * pad - fw) // stride + 1

    # 增加填充
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')

    # 构造中间变量col，是一个6维数组(N, C, fh, fw, out_h, out_w)
    col = np.zeros((N, C, fh, fw, out_h, out_w))

    for y in range(fh):
        y_max = y + stride * out_h # y可以向下滑动(out_h - 1)次，这里用 y : y + y_max : stride来表示这个滑动过程
        for x in range(fw):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y : y_max : stride, x : x_max : stride]

    # 将6维的col转化成二维的
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col

def col2im(col, input_shape, fh, fw, stride = 1, pad = 0):
    """

    :param col: 反向传播过程中的二维矩阵，结构为(N * out_h, out_w, C * fh * fw)
    :param input_shape: 正向传播过程中输入的四维数组的shape，结构为(N, C, H, W)
    :param fh: 滤波器高
    :param fw: 滤波器宽
    :param stride: 步幅
    :param pad: 填充行/列数

    :return: 四维矩阵(N, C, H, W)
    """
    N, C, H, W = input_shape

    # 计算输出的out_h和out_w
    out_h = (H + 2 * pad - fh) // stride + 1
    out_w = (W + 2 * pad - fw) // stride + 1

    # 将二维的col变成6维的col，即(N, C, fh, fw, out_h, out_w)
    col = col.reshape(N, out_h, out_w, C, fh, fw).transpose(0, 3, 4, 5, 1, 2)

    # 初始化四维对象img，结构为(N, C, H, W)
    img = np.pad(np.zeros(input_shape), [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')

    # 构造同等的四维矩阵，每格元素存放当前位置元素相加的次数，因为会重复相加
    cnt = np.zeros_like(img)

    for y in range(fh):
        y_max = y + stride * out_h
        for x in range(fw):
            x_max = x + stride * out_w
            img[:, :, y : y_max : stride, x : x_max : stride] += col[:, :, y, x, :, :]
            cnt[:, :, y : y_max : stride, x : x_max : stride] = cnt[:, :, y : y_max : stride, x : x_max : stride] + 1 # 对应位置+1

    col = img / cnt # 对应位置除以相加次数，每个位置至少相加一次，因此不会出现除以0的情况
    return col[:, :, pad : pad + H, pad : pad + W] # 返回的时候填充不要算进去



