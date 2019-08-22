# !user/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from common.optimizer import *

class Trainer:
    """进行神经网络训练的类

    """

    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs = 20, mini_batch_num = 100, optimizer = 'SGD',
                 optimizer_params = None):
        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.mini_batch_num = mini_batch_num
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params

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

        self.current_iter += 1

        # 随机选择mini-batch
        batch_mask = np.random.choice(self.train_size, self.mini_batch_num)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        # 前向、反向传播，获取梯度
        grads = self.network.gradient(x_batch, t_batch, use_weight_decay = self.network.use_weight_decay)

        # 更新梯度
        self.optimizer.update(self.network.params, grads)

        # 保存每一轮的loss
        loss = self.network.loss(x_batch, t_batch, for_trainning = True, use_weight_decay = self.network.use_weight_decay)
        self.train_loss_list.append(loss)

        # 保存每一epoch的accuracy
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            train_acc = self.network.accuracy(self.x_train, self.t_train)
            self.train_acc_list.append(train_acc)
            test_acc = self.network.accuracy(self.x_test, self.t_test)
            self.test_acc_list.append(test_acc)

    def train(self):
        for i in range(self.max_iter_num):
            self.train_step()