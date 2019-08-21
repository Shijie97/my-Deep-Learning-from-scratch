# !user/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from Five_Layer_Network import Five_Layer_Network

# 本代码旨在验证权重初始值对训练效果的影响
# 层数为5，每层神经元数量为100
# 初始权重设定为 标准差为1和0.01的高斯分布，以及Xavier初值和He初值
# 迭代次数仅一轮
# 输入数据的批量为1000

# 设置超参数
node_num = 100
hidden_layer_size = 5
x = np.random.randn(1000, 100)
w_type = ['std_1', 'std_0.01', 'Xavier', 'He']
f_type = ['sigmoid', 'sigmoid', 'tanh', 'ReLu']
network_list = []

for i in range(4):
    network = Five_Layer_Network(node_num = node_num,
                                 hidden_layer_size = hidden_layer_size,
                                 weight_distribution_type = w_type[i],
                                 activation_function_type = f_type[i])
    network_list.append(network)

for index, network in enumerate(network_list):
    network.forward(x)
    network.plot_hist(index)

plt.show()