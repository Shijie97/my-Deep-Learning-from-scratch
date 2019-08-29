# !user/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from common.functions import *
import warnings
from common.util import *
warnings.filterwarnings('error')

class ReLu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0) # 生成布尔数组
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
        self.original_x_shape = None

    def forward(self, x):
        # 这里输入的x的结构不一定是(N, XXX)的，N为批量数，所以要进行一下转换
        # 一般传进来的，第一维都是N
        # 典型的例子就是Pooling -> Affine这个例子，传进来的是四维的信息流，但是Affine只接受二维的，所以要转化
        self.original_x_shape = x.shape
        self.x = x.reshape(self.original_x_shape[0], -1)
        return np.dot(self.x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)

        # 反向的时候还要把dx还原成传进来的张量形式
        dx = dx.reshape(self.original_x_shape)
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

class Convolution:
    def __init__(self, W, b, stride = 1, pad = 0):

        """

        :param W: 滤波器参数，结构为(FN, C, fh, fw)
        :param b: 偏置参数， 结构为(1, FN)，这里不存在N的问题，因为每个批量的b都是一样的，FN列的每一列都代表对应通道的偏置
        :param stride: 步幅
        :param pad: 填充
        """

        self.W = W
        self.b = b
        self.stride = stride
        self.pad =pad

        # 梯度
        self.dW = None
        self.db = None

        # 反向传播时要用到的中间数据
        self.x = None
        self.col = None # 输入的x从四维变成二维的col
        self.col_W = None # 四维的滤波器W变成二维的col_W

    def forward(self, x):

        """

        :param x: 输入的四维矩阵，结构为(N, C ,H, W)
        :return: 输出也是个四维矩阵，结构为(N, FN / C, HH / out_h, WW / out_w) 输出的通道数取决于滤波器的数量，即W的数量宽高取决于out_h和out_w
        """

        N, C, H, W = x.shape
        FN, C, fh, fw = self.W.shape

        # 计算out_h和out_w，计算这个是因为要将二维的out转化为四维的，然而二维的out有N * out_h * out_w行，因此必须知道这两个参数
        out_h = int((H + 2 * self.pad - fh) // self.stride) + 1
        out_w = int((W + 2 * self.pad - fw) // self.stride) + 1

        # x从四维变成二维
        self.col = im2col(x, fh, fw, stride = self.stride, pad = self.pad)

        # W从四维变成二维
        self.col_W = self.W.transpose(1, 2, 3, 0).reshape(-1, FN)

        out = np.dot(self.col, self.col_W) + self.b

        # 将二维的out转化为四维的输出
        out = out.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)

        self.x = x # 保存一下x用于反向梯度
        return out

    def backward(self, dout):
        """

        :param dout: 四维数组结构为(N, FN / C, HH / out_h, WW / out_w)，不再赘述
        :return: 四维数组dx，结构为(N, C ,H, W)
        """
        FN, C, fh, fw = self.W.shape

        # dout转化为二维数组
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        # 根据Affine层的规则反向求梯度
        dx = np.dot(dout, self.col_W.T)
        self.dW = np.dot(self.col.T, dout)
        self.db = np.sum(dout, axis = 0)

        # 将dx转化成四维的
        dx = col2im(dx, self.x.shape, fh, fw, stride = self.stride, pad = self.pad)

        # 将dW转化成四维的
        self.dW = self.dW.reshape((C, fh, fw, FN)).transpose(3, 0, 1, 2)

        return dx

class Pooling:
    def __init__(self, ph, pw, stride = 1, pad = 0):
        """

        :param ph: 池化层滤波器高
        :param pw: 池化层滤波器宽
        :param stride: 步幅
        :param pad: 填充行/列数，池化层的计算不需要填充，这里的pad是给im2col和col2im提供参数
        """
        self.ph = ph
        self.pw = pw
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        """

        :param x: 输入的四维数据(N, C, H, W)
        :return: 池化后的四维数组
        """

        N, C, H, W = x.shape

        # 计算out_h和out_w，此过程没有pad参与
        out_h = int((H - self.ph) // self.stride) + 1
        out_w = int((W - self.pw) // self.stride) + 1

        # 先用im2col函数将x转化成二维的，结构为(N * out_h * out_w, C * ph * pw)
        col = im2col(x, self.ph, self.pw, stride = self.stride, pad = self.pad)

        # 现在的x结构为二维(N * out_h * out_w, C * ph * pw)
        # 再将其变成二维(N * out_h * out_w * C, ph * pw)，这样每行就是ph * pw这么多个元素了
        col = col.reshape(-1, self.ph * self.pw)

        # 记录每行最大值的索引
        self.arg_max = np.argmax(col, axis = 1)

        # 求每行的最大值，此时x的结构为(N * out_h * out_w * C, 1)，即只有一列
        col = np.max(col, axis = 1)

        # x转化成最终输出的四维结果
        dout = col.reshape((N, out_h, out_w, C)).transpose(0, 3, 1, 2)
        self.x = x

        return dout

    def backward(self, dout):
        """

        :param dout: 池化后的四维数组，结构为(N, C, HH / out_h, WW / out_w)，宽高取决于out_h和out_w
        :return: 池化前的四维数组dx，结构为(N, C, H, W)
        """
        # dout的结构为(N, C, out_h, out_w)，但是在计算argmax的时候，结构为(N * out_h * out_w * C, 1)，所以要先转换一下
        dout = dout.transpose(0, 2, 3, 1)
        # 初始化展开后的二维数组(N * out_h * out_w * C, ph * pw)，亦即(dout.size, self.ph * self.pw)
        dmax = np.zeros((dout.size, self.ph * self.pw))
        # 每一行都填上最大值
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        # dmax现在是二维，结构为(N * out_h * out_w * C, ph * pw)
        # 要使用col2im，就必须变成(N * out_h * out_w, C * ph * pw)这种结构
        # 正好dout的shape为(N, out_h, out_w, C)
        dmax = dmax.reshape(dout.shape[0] * dout.shape[1] * dout.shape[2], -1)
        dx = col2im(dmax, self.x.shape, self.ph, self.pw, stride = self.stride, pad = self.pad)
        return dx
