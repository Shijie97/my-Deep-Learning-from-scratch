# !user/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import os, sys
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from ch05.two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt
from common.optimizer import *
from collections import  OrderedDict

# 读入数据

(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)

network = {}

iters_num = 10000 # 迭代轮数
train_size = x_train.shape[0]
batch_size = 100 # 小批量数
learning_rate = 0.01

train_loss_list = {}
index_list = []

optimizer_dict = OrderedDict()
optimizer_dict['SGD'] = SGD()
optimizer_dict['Momentum'] = Momentum()
optimizer_dict['AdaGrad'] = AdaGrad()
optimizer_dict['RMSProp'] = RMSprop()
optimizer_dict['Adam'] = Adam()

for key in optimizer_dict.keys():
    train_loss_list[key] =[]
    network[key] = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iter_per_epoch = max(train_size / batch_size, 1) # 一个epoch需要遍历的轮数

for i in range(0, iters_num):

    print('这是第%d轮' % i)

    # mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key, optimizer in optimizer_dict.items():
        grads = network[key].gradient(x_batch, t_batch)
        optimizer.update(network[key].params, grads)
        loss =network[key].loss(x_batch, t_batch)
        if i % 100 == 0:
            train_loss_list[key].append(loss)
            if key == list(optimizer_dict.keys())[0]:
                index_list.append(i)

# 画损失函数
markers = {'SGD':'o', 'Momentum':'x', 'AdaGrad':'s', 'RMSProp': 'D', 'Adam':'+'}
for key in markers.keys():
    plt.plot(index_list, train_loss_list[key], label = key, markevery = 500, marker = markers[key])
plt.xlabel = 'round'
plt.ylabel = 'loss'
plt.legend(loc = 'upper right')
plt.ylim(0, 2.5)
plt.show()




