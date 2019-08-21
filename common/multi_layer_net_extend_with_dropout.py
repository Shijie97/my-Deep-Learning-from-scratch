# !user/bin/python
# -*- coding: UTF-8 -*-

from common.multi_layer_net_extend import MultiLayerNetExtend
from common.layers import *

class MultiLayerNetExtend_With_Dropout(MultiLayerNetExtend):

    def __init__(self,
                 input_size,
                 hidden_size_list,
                 output_size,
                 activation = 'ReLu',
                 weight_init_std = 'ReLu',
                 weight_decay_lambda = 0,
                 use_BatchNormalization = False,
                 use_weight_decay = False,
                 use_dropout = False,
                 dropout_version = 'inverted',
                 dropout_rate = 0.5
                 ):
        self.use_dropout = use_dropout
        self.dropout_version = dropout_version
        self.dropout_rate = dropout_rate
        super().__init__(input_size,
                       hidden_size_list,
                       output_size,
                       activation,
                       weight_init_std,
                       weight_decay_lambda,
                       use_BatchNormalization,
                       use_weight_decay
                       )

    def construct_layer_except_last(self):
        activaion_layer = {'Sigmoid': Sigmoid, 'ReLu': ReLu}  # value存的是函数
        for idx in range(1, len(self.hidden_size_list) + 1):
            # Affine层
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            # 将BN层放在激活函数之前
            if self.use_BatchNormalization:
                self.params['gamma' + str(idx)] = np.ones(self.hidden_size_list[idx - 1])
                self.params['beta' + str(idx)] = np.zeros(self.hidden_size_list[idx - 1])
                self.layers['BatchNormalization' + str(idx)] = BatchNormalization_For_FNN(
                    self.params['gamma' + str(idx)],
                    self.params['beta' + str(idx)])
            # 激活函数层
            self.layers['Activation_Function' + str(idx)] = activaion_layer[self.activation]()

            # dropout放在激活函数之后
            if self.use_dropout:
                if self.dropout_version == 'inverted':
                    self.layers['Dropout' + str(idx)] = Dropout_Inverted_Version(self.dropout_rate)
                elif self.dropout_version == 'vanilla':
                    self.layers['Dropout' + str(idx)] = Dropout_Vanilla_Version(self.dropout_rate)

    def predict(self, x, for_trainning):
        for layer_name, layer in self.layers.items():
            if 'BatchNormalization' in layer_name or 'Dropout' in layer_name:
                x = layer.forward(x, for_trainning)
            else:
                x = layer.forward(x)

        return x