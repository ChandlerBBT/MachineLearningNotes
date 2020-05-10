# encoding:utf-8
# project: MachineLearningNotes
# author: chandler
# file: nn_single_layer.py
# create_time: 2020/5/10
# IDE: Pycharm


import numpy as np
import json
import matplotlib.pyplot as plt
from NeuralNetworks.NN.nn_single_layer import Network


def load_data():
    # 从文件导入数据
    datafile = r"C:\Users\Administrator\PycharmProjects\MachineLearningNotes\DataSets\housing.data"
    data = np.fromfile(datafile, sep=' ')
    print(data.shape)
    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                      'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    feature_num = len(feature_names)

    # 将原始数据进行Reshape，变成[N, 14]这样的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])

    # 将原数据集拆分成训练集和测试集
    # 这里使用80%的数据做训练，20%的数据做测试
    # 测试集和训练集必须是没有交集的
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    # 计算train数据集的最大值，最小值，平均值
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), \
                                 training_data.sum(axis=0) / training_data.shape[0]

    # 对数据进行归一化处理
    for i in range(feature_num):
        #print(maximums[i], minimums[i], avgs[i])
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data


if __name__ == "__main__":
    # 获取数据
    train_data, test_data = load_data()

    # 创建网络
    net = Network(13)
    # 启动训练
    losses = net.train(train_data, num_epoches=50, batch_size=100, eta=0.1)

    # 画出损失函数的变化趋势
    plot_x = np.arange(len(losses))
    plot_y = np.array(losses)
    plt.plot(plot_x, plot_y)
    plt.show()