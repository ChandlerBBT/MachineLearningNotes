# encoding:utf-8
# project: MachineLearningNotes
# author: chandler
# file: network.py
# create_time: 2020/4/26
# IDE: Pycharm

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class Network(object):
    def __init__(self, sizes):
        """

        :param sizes:列表sizes包含各层神经元的数量
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.bias = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

