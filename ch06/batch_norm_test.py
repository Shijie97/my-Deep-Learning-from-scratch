# !user/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from common.layers import *
from common.optimizer import *
from common.multi_layer_net_extend import MultiLayerNetExtend
from dataset.mnist import load_mnist
import warnings

warnings.simplefilter('ignore', ResourceWarning)

(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)

# 只选前1000个样本作为少量样本
x_train, t_train = x_train[:1000], t_train[:1000]

# 固定参数
train_size = x_train.shape[0]
learning_rate = 0.01
batch_size = 100
iter_num_per_epch = max(1, train_size / batch_size)
iter_num = 1000000000
epoch_cnt_max = 20

# 有几个网络就要有几个Momentum，因为要初始化N次
optimizer = Momentum(lr = learning_rate, momentum = 0.99)
optimizer_BN = Momentum(lr = learning_rate, momentum = 0.99)
# SGD仅仅初始化lr，所以只用一个就行
optimizer_SGD = SGD(lr = learning_rate)

def __trainning(weight_init_std, index):


    epoch_num = 0
    acc_train_list = []
    acc_train_BN_list = []

    input_size = 784
    hidden_size_list = [100] * 5
    output_size = 10
    activation = 'ReLu'
    weight_decay_lambda = 1

    network = MultiLayerNetExtend(input_size = input_size,
                                  hidden_size_list = hidden_size_list,
                                  output_size = output_size,
                                  activation = activation,
                                  weight_init_std = weight_init_std,
                                  weight_decay_lambda = weight_decay_lambda,
                                  use_BatchNormalization = False,
                                  use_weight_decay = False)
    network_BN = MultiLayerNetExtend(input_size = 784,
                                  hidden_size_list = hidden_size_list,
                                  output_size = output_size,
                                  activation = activation,
                                  weight_init_std = weight_init_std,
                                  weight_decay_lambda = weight_decay_lambda,
                                  use_BatchNormalization = True,
                                  use_weight_decay = False)


    # 开始训练，训练分下面几步
    # 1. 从样本中随机挑选出batch_size个样本
    # 2. 前向传播、反向传播、获取梯度
    # 3. 更新梯度
    # 4. 重复1 ~ 2直至循环结束
    for i in range(iter_num):

        print('现在是W' + str(index) + '的第' + str(i) + '轮循环')

        # 挑选mini_batch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 获取梯度，更新梯度
        for network_ in (network, network_BN):
            grads = network_.gradient(x_train, t_train, use_weight_decay = False)
            # optimizer_SGD.update(params = network_.params, grads = grads)
            if network_ == network:
                optimizer.update(params=network_.params, grads=grads)
            else:
                optimizer_BN.update(params=network_.params, grads=grads)

        # 每一个epoch就计算两个网络的精度然后装进数组
        if i % iter_num_per_epch == 0:
            network_acc = network.accuracy(x_batch, t_batch)
            network_BN_acc = network_BN.accuracy(x_batch, t_batch)
            acc_train_list.append(network_acc)
            acc_train_BN_list.append(network_BN_acc)

            epoch_num += 1

            if epoch_num >= epoch_cnt_max:
                break

    return acc_train_list, acc_train_BN_list

# 一共16次试验，对应16个W，从1到1e-4，成等比数列排布，共16个
w_list = np.logspace(0, -4, 16)
epoch_x = np.arange(epoch_cnt_max)
# acc_train_list, acc_train_BN_list = __trainning(w_list[0], 1)
# plt.plot(epoch_x, acc_train_list)
# plt.plot(epoch_x, acc_train_BN_list, linestyle = '--')


for i in range(16):
    print('=' * 20 + '现在是第' + str(i + 1) + '个W' + '=' * 20)
    acc_train_list, acc_train_BN_list = __trainning(w_list[i], i + 1)
    plt.subplot(4, 4, i + 1)
    plt.title('W' + str(i + 1) + ': ' + str(round(w_list[i], 5)))
    if i != 15:
        plt.plot(epoch_x, acc_train_list)
        plt.plot(epoch_x, acc_train_BN_list, linestyle = '--')
    else:
        plt.plot(epoch_x, acc_train_list, label = 'without_BN')
        plt.plot(epoch_x, acc_train_BN_list, linestyle='--', label = 'with_BN')
        # plt.legend(loc = 'upper right')

    if i % 4 != 0:
        plt.yticks([])
    else:
        plt.ylabel('accuracy')

    if i < 12:
        plt.xticks([])
    else:
        plt.xlabel('epoch')
        plt.xticks(np.arange(0, epoch_cnt_max, 5))

    plt.ylim(0, 1)

plt.show()