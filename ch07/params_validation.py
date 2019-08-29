# !user/bin/python
# -*- coding: UTF-8 -*-

from dataset.mnist import load_mnist
from simple_convnet import SimpleConvnet

(x_train, t_train), (x_test, t_test) = load_mnist(flatten = False, normalize = True, one_hot_label = True)

network = SimpleConvnet(input_dim = (1, 28, 28), conv_params = {'filter_num' : 30, 'filter_size' : 5,
                                'stride' : 1, 'pad' : 0}, hidden_size = 100, output_size = 10, weight_init_std = 'ReLu')
network.load_params()

acc_test = network.accuracy(x_test, t_test)
print('accuracy: ' + str(acc_test)) # accuracy: 0.8491