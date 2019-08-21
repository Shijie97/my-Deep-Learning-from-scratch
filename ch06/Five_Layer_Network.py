# !user/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class Five_Layer_Network:
    def __init__(self, node_num, hidden_layer_size, weight_distribution_type, activation_function_type):
        self.node_num = node_num
        self.hidden_layer_size = hidden_layer_size
        self.activations = None
        self.weight_distribution_type = weight_distribution_type
        self.activation_function_type = activation_function_type
        self.w = None
        self.activation_function = None

    def get_weight(self):
        if self.weight_distribution_type == 'std_1':
            return np.random.randn(self.node_num, self.node_num) * 1
        elif self.weight_distribution_type == 'std_0.01':
            return np.random.randn(self.node_num, self.node_num) * 0.01
        elif self.weight_distribution_type == 'Xavier':
            return np.random.randn(self.node_num, self.node_num) * np.sqrt(1.0 / self.node_num)
        elif self.weight_distribution_type == 'He':
            return np.random.randn(self.node_num, self.node_num) * np.sqrt(2.0 / self.node_num)

    def get_activation_function(self):
        if self.activation_function_type == 'ReLu':
            return lambda x : np.maximum(0, x)
        elif self.activation_function_type == 'sigmoid':
            return lambda x : 1 / (1 + np.exp(-x))
        elif self.activation_function_type == 'tanh':
            return lambda x : np.tanh(x)

    def forward(self, x):
        if self.activations is None:
            self.activations = {}

        self.w = self.get_weight()
        self.activation_function = self.get_activation_function()

        for i in range(self.hidden_layer_size):
            if i != 0:
                x = self.activations[i - 1]

            z = np.dot(x, self.w) # 此时a是1000 * 100，共计100000个数字
            a = self.activation_function(z) # 此时的z也是1000 * 100，共计100000个数字
            self.activations[i] = a

    def plot_hist(self, index):
        for i, a in self.activations.items():
            plt.subplot(4, len(self.activations), index * len(self.activations) + i + 1)
            plt.title(str(i + 1) + '-layer')
            if i != 0:
                plt.yticks([], [])
            plt.hist(a.flatten(), bins = 30, range = (0, 1))
            plt.ylim(0, 10000)
