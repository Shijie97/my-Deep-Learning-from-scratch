# !user/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import os, sys
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from ch05.two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt

# 读入数据

(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)

# 构造神经网络对象
network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)

iters_num = 10000 # 迭代轮数
train_size = x_train.shape[0]
batch_size = 100 # 小批量数
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = [] # 最重要的评价指标

iter_per_epoch = max(train_size / batch_size, 1) # 一个epoch需要遍历的轮数

for i in range(0, iters_num):

    print('这是第%d轮' % i)

    # mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 求梯度
    # 这个过程包括了前向传播，反向传播，以及获取梯度三个过程
    grads = network.gradient(x_batch, t_batch)

    # 更新梯度
    for key in ['W1', 'b1', 'W2', 'b2']:
        network.params[key] -= learning_rate * grads[key]

    # 求误差
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('这是第', i, '轮，', '训练样本准确率：', train_acc, '测试样本准确率：', test_acc)

print('训练集的精度为', train_acc_list[-1])
print('测试集的精度为', test_acc_list[-1])

# 循环结束，绘制图像
x = np.arange(len(train_acc_list))
plt.xlabel = 'epochs'
plt.ylabel = 'accuracy'
plt.ylim(0, 1.0)
plt.plot(x, train_acc_list, label = 'train_acc')
plt.plot(x, test_acc_list, label = 'test_acc', linestyle = '--')
plt.legend(loc = 'lower right')
plt.show()

# 画损失函数
x = np.arange(iters_num)
plt.plot(x, train_loss_list, label = 'train_loss_list')
plt.xlabel = 'round'
plt.ylabel = 'loss'
plt.legend(loc = 'upper right')
plt.ylim(0, 2.5)
plt.show()




