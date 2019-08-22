# !user/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from common.trainer import Trainer
from common.util import *
from common.multi_layer_net_extend import MultiLayerNetExtend
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)

# 选择少量数据以加速
x_train = x_train[:400]
t_train = t_train[:400]

# 分割验证数据
validation_rate = 0.2
validation_num = int(x_train.shape[0] * validation_rate)
x_train, x_test = shuffle_dataset(x_train, x_test) # 打乱数据

# 验证集
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]

#训练集
x_train = x_train[:validation_num]
t_train = t_train[:validation_num]

def __train(lr, weight_decay_lambda, epoch_num = 120):
    network = MultiLayerNetExtend(input_size = 784, hidden_size_list = [100] * 5,
                                  output_size = 10, activation = 'ReLu', weight_init_std = 'ReLu',
                                  weight_decay_lambda = weight_decay_lambda, use_BatchNormalization = False,
                                  use_weight_decay = True)
    # 注意下面传入的不是测试集而是验证集
    trainer = Trainer(network = network, x_train = x_train, t_train = t_train, x_test = x_val,
                      t_test = t_val, epochs = epoch_num, mini_batch_num = 100, optimizer = 'SGD',
                      optimizer_params = {'lr' : lr})
    trainer.train()
    return trainer.train_acc_list, trainer.test_acc_list # 返回一次实验测试集和验证集的精度

# 超参数随机搜索
loop_num = 100 # 尝试次数
res_val = {} # 将对应lr和权重的实验验证集数组结果藏在这个字典里，下同
res_train = {}

for i in range(loop_num):

    print('现在是第' + str(i + 1) + '轮训练')

    # 选择lr和decay的范围
    lr = 10 ** np.random.uniform(-5, -2)
    weight_decay_lambda = 10 ** np.random.uniform(-6, 0)

    # 开始训练并得到结果
    val_acc_list, train_acc_list = __train(lr = lr, weight_decay_lambda = weight_decay_lambda)
    key = 'lr = ' + str(round(lr, 5)) + ', weight_decay_lambda = ' + str(round(weight_decay_lambda, 5)) # 保留5位小数
    res_val[key] = val_acc_list
    res_train[key] = train_acc_list

# 画图
graph_num = 20
graph_col = 5
graph_row = int(np.ceil(graph_num / graph_col))

# 在100次实验中，挑出前20个验证集精度最高的
# 对字典进行排序，lambda表达式中，x[1]表示对value进行考察，x[1][-1]表示对value的最后一个元素，即最后一轮的精度，reverse = True表示降序
res_val = sorted(res_val.items(), key = lambda x : x[1][-1], reverse = True) # 返回一个数组，装有键值对，键值对之间用元组的形式呈现

for i in range(graph_num):
    plt.subplot(graph_row, graph_col, i + 1)
    val_acc_list = res_val[i][1] # 元组的第二项，即精度数组
    key = res_val[i][0] # 元组的第一项，即key
    train_acc_list = res_train[key]
    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list, label = 'val_acc_list', color = 'red')
    plt.plot(x, train_acc_list, linestyle = '--', label = 'train_acc_list', color = 'blue')

    # if i % graph_col == 0:
    #     plt.ylabel('accuracy')
    # else:
    #     plt.yticks([])
    #
    # if i >= (graph_row - 1) * graph_col:
    #    plt.xlabel('epoch')
    # else:
    #     plt.xticks([])

    # plt.title('Best top' + str(i + 1))
    # plt.legend(loc = 'upper right')
    plt.ylim(0, 1)

    print('现在i = ' + str(i + 1) + ', 参数为 ' + key)

plt.show()
