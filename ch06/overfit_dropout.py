# !user/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from common.multi_layer_net_extend_with_dropout import MultiLayerNetExtend_With_Dropout
from common.optimizer import *
from common.layers import *
from dataset.mnist import load_mnist
import warnings

warnings.simplefilter('ignore', ResourceWarning)

(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)

# 为了导致过拟合，选取更少的样本
x_train, t_train = x_train[:300], t_train[:300]

# 固定参数
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01
iter_num_per_epoch = max(1, train_size / batch_size)
iter_num = 1000000000
epoch_cnt_max = 30 # epoch允许的最大值

# 生成优化器
optimizer = SGD(lr = learning_rate)

# 网络初始化参数
input_size = 784
hidden_size_list = [100] * 5
output_size = 10
activation = 'ReLu' # 每一层之间的激活函数
weight_init_std = 'ReLu' # 根据激活函数以初始化权值分布
weight_decay_lambda = 1
epoch_num = 0

train_acc_list = []
test_acc_list = []
train_acc_dropout_list = []
test_acc_dropout_list = []

# 构造网络对象
network = MultiLayerNetExtend_With_Dropout(input_size = input_size,
                              hidden_size_list = hidden_size_list,
                              output_size = output_size,
                              activation = activation,
                              weight_init_std = weight_init_std,
                              weight_decay_lambda = weight_decay_lambda,
                              use_BatchNormalization = False,
                              use_weight_decay = False,
                              use_dropout = False,
                              dropout_version = 'inverted',
                              dropout_rate = 0.2
                              )
network_dropout = MultiLayerNetExtend_With_Dropout(input_size = input_size,
                              hidden_size_list = hidden_size_list,
                              output_size = output_size,
                              activation = activation,
                              weight_init_std = weight_init_std,
                              weight_decay_lambda = weight_decay_lambda,
                              use_BatchNormalization = False,
                              use_weight_decay = False,
                              use_dropout = True,
                              dropout_version = 'inverted',
                              dropout_rate = 0.2
                              )

# 开始训练，训练分下面几步
# 1. 从样本中随机挑选出batch_size个样本
# 2. 前向传播、反向传播、获取梯度
# 3. 更新梯度
# 4. 重复1 ~ 2直至循环结束

for i in range(iter_num):

    print('现在是第' + str(i) + '轮')

    # 随机挑选出batch_size个样本
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 前向传播，反向传播，获取梯度
    for _network in (network, network_dropout):
        grads = _network.gradient(x_batch, t_batch, use_weight_decay = _network.use_weight_decay)
        optimizer.update(_network.params, grads)

    if i % iter_num_per_epoch == 0:
        epoch_num += 1

        train_acc_network = network.accuracy(x_train, t_train)
        train_acc_list.append(train_acc_network)

        test_acc_network = network.accuracy(x_test, t_test)
        test_acc_list.append(train_acc_network)


        train_acc_network_dropout = network_dropout.accuracy(x_train, t_train)
        train_acc_dropout_list.append(train_acc_network_dropout)

        test_acc_network_dropout = network_dropout.accuracy(x_test, t_test)
        test_acc_dropout_list.append(test_acc_network_dropout)

        if epoch_num >= epoch_cnt_max:
            break

x = np.arange(epoch_cnt_max)
plt.plot(x, train_acc_list, label = 'without dropout for train', marker = 'o')
plt.plot(x, train_acc_dropout_list, label = 'with dropout for train', marker = 's')
plt.plot(x, test_acc_list, label = 'without dropout for test', marker = 'v', linestyle = '--')
plt.plot(x, test_acc_dropout_list, label = 'with dropout for test', marker = '^', linestyle = '--')
plt.ylim(0, 1)
plt.title('compare between without dropout and with dropout')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(loc = 'upper right')
plt.show()