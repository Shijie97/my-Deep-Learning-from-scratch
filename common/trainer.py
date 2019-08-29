# !user/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from common.optimizer import *

class Trainer:
    """进行神经网络训练的类

    """

    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs = 20, mini_batch_num = 100, optimizer = 'SGD',
                 optimizer_params = None, num_of_sample_per_epoch = None):
        # num_of_sample_per_epoch为每个epoch求精度时参与的样本数

        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.mini_batch_num = mini_batch_num
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.num_of_sample_per_epoch = num_of_sample_per_epoch

        self.optimizer = eval(self.optimizer)(**self.optimizer_params)
        self.train_size = self.x_train.shape[0]
        self.iter_per_epoch = max(1, self.train_size / self.mini_batch_num)
        self.max_iter_num = int(self.epochs * self.iter_per_epoch)

        self.current_iter = 0
        self.current_epoch = 0

        self.train_loss_list = [] # 取每一轮
        self.train_acc_list = [] # 取每一epoch
        self.test_acc_list = [] # 取每一epoch

    # 每一轮的操作
    def train_step(self):
        print('当前的epoch为：' + str(self.current_epoch) + ', 此epoch当前的轮数为：' + str(self.current_iter) )

        self.current_iter += 1

        # 随机选择mini-batch
        batch_mask = np.random.choice(self.train_size, self.mini_batch_num)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        # 前向、反向传播，获取梯度
        grads = self.network.gradient(x_batch, t_batch)

        # print('-' * 10)
        # print(self.network.params['W1'][0])
        # 更新梯度
        self.optimizer.update(self.network.params, grads)
        # print(self.network.params['W1'][0])
        # print('-' * 10)

        # 保存每一轮的loss
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)

        # 保存每一epoch的accuracy
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            sample_x_train, sample_t_train = self.x_train, self.t_train
            sample_x_test, sample_t_test = self.x_test, self.t_test

            if not self.num_of_sample_per_epoch is None:
                num = self.num_of_sample_per_epoch
                sample_x_train = self.x_train[:num]
                sample_t_train = self.t_train[:num]
                sample_x_test = self.x_test[:num]
                sample_t_test = self.t_test[:num]

            train_acc = self.network.accuracy(sample_x_train, sample_t_train)
            self.train_acc_list.append(train_acc)
            test_acc = self.network.accuracy(sample_x_test, sample_t_test)
            self.test_acc_list.append(test_acc)
            print('train_acc: ' + str(train_acc) + ', test_acc: ' + str(test_acc))

    def train(self):
        for i in range(self.max_iter_num):
            self.train_step()