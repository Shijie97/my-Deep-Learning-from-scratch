# !user/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.trainer import Trainer
from simple_convnet import SimpleConvnet

# 注意，这里的(x_train, t_train), (x_test, t_test)都不是flatten的，后面求要格外注意
(x_train, t_train), (x_test, t_test) = load_mnist(flatten = False, normalize = True, one_hot_label = True)


x_train = x_train[:1000]
t_train = t_train[:1000]

network = SimpleConvnet(input_dim = (1, 28, 28), conv_params = {'filter_num' : 30, 'filter_size' : 5,
                                'stride' : 1, 'pad' : 0}, hidden_size = 100, output_size = 10, weight_init_std = 'ReLu')
epochs = 250
mini_batch_num = 100
optimizer = 'SGD'
optimizer_params = {'lr' : 0.01}
num_of_sample_per_epoch = 2000

trainer = Trainer(network, x_train, t_train, x_test, t_test, epochs, mini_batch_num, optimizer, optimizer_params, num_of_sample_per_epoch)
trainer.train()

# 保存参数
network.save_params()
print('Save Successfully!')

# 画图
plt.figure(1)
x1 = np.arange(epochs)
plt.plot(x1, trainer.train_acc_list, marker = 'o', markevery = 2, color = 'red', label = 'train')
plt.plot(x1, trainer.test_acc_list, marker = 's', markevery = 2, color = 'blue', label = 'test', linestyle = '--')
plt.ylim(0, 1)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(loc = 'upper right')


plt.figure(2)
x2 = np.arange(trainer.max_iter_num)
plt.plot(x2, trainer.train_loss_list, label = 'loss', color = 'red')
plt.ylim(0, 3)
plt.xlabel('iter')
plt.ylabel('loss')
plt.legend(loc = 'upper right')
plt.show()


