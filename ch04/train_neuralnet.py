# !user/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import sys, os
sys.path.append(os.pardir)
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = [] # 每轮迭代后的损失
train_acc_list = [] # 每轮epoch，训练样本的精度
test_acc_list = [] # 每轮epoch，测试样本的精度，这是为了防止过拟合，提高泛化能力

# 超参数
iters_num = 1000 # 循环次数
train_size = x_train.shape[0]
batch_size = 100 # 小批量数，即有几个epoch
batch_num = int(train_size / batch_size) # 每个epoch迭代的次数
learning_rate = 0.1

# 构造网络对象
network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)

# 开始循环
# 每轮循环，要干的事如下
# 1. 从10000个样本里面随机挑100个出来，包含100个训练的和100个监督的
# 2. 计算各个矩阵的梯度
# 3. 矩阵更新，计算精度
# 4. 重复1 ~ 3直至循环结束

for i in range(iters_num):
    print('这是第%d轮' % i)

    # 随机挑100个
    random_array = np.random.choice(train_size, batch_size) # 返回一个数组
    x_batch = x_train[random_array]
    t_batch = t_train[random_array]

    # 计算矩阵的梯度
    grads = network.numerical_gradient_in_class(x_batch, t_batch)
    # grads = network.gradient(x_batch, t_batch)
    print(np.sum(grads['W1']))


    # 更新矩阵
    for x in ('W1', 'b1', 'W2', 'b2'):
        network.params[x] -= learning_rate * grads[x]

    # 计算损失这一轮的损失并加到train_loss_list里
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 判断是不是到了一个epoch，如果到了，更新相应的数组
    if i % batch_num == 0:
        train_acc = network.accuracy(x_train, t_train) # 计算所有训练样本的准确率
        test_acc = network.accuracy(x_test, t_test) # 计算所有测试样本的准确率
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('这是第', i ,'轮，', '训练样本准确率：', train_acc, '测试样本准确率：', test_acc)

# 循环结束，绘制图像
x = np.arange(len(train_acc_list))
plt.xlabel = 'epochs'
plt.ylabel = 'accuracy'
plt.legend(loc = 'lower right')
plt.ylim(0, 1.0)
plt.plot(x, train_acc_list, label = 'train_acc')
plt.plot(x, test_acc_list, label = 'test_acc', linestyle = '--')
plt.show()

# 画损失函数
x = np.arange(iters_num)
plt.plot(x, train_loss_list, label = 'train_loss_list')
plt.xlabel = 'round'
plt.ylabel = 'loss'
plt.legend(loc = 'upper right')
plt.ylim(0, 1.0)
plt.show()