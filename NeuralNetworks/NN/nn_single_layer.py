# encoding:utf-8
# project: MachineLearningNotes
# author: chandler
# file: nn_single_layer.py
# create_time: 2020/5/10
# IDE: Pycharm


import numpy as np


class Network(object):
    def __init__(self, num_of_weights):
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.

    def forward(self, x):
        """
        前向传播
        :param x: 样本集 
        :return: 向量，(m, 1) m表示样本的数量
        注意，x-(m,num_of_weights), self.w-(num_of_weights,1) ==> (m,1)
        """
        return np.dot(x, self.w) + self.b
    
    def loss(self, forward_result, y):
        """
        计算损失
        :param forward_result: 前向传播过程计算得到的向量
        :param y: 样本的标签向量 - (m,1)
        :return: 数值
        """
        error = forward_result - y
        cost = error * error
        return np.sum(cost) / len(error)

    def gradient(self, x, y):
        """
        计算梯度，值得注意的是，梯度的值完全取决于样本
        :param x:
        :param y:
        :return:
        """
        z = self.forward(x)
        N = len(x)
        # z-y 是一个(m,1)的向量，表示每一个样本产生的误差，用每个样本产生的误差*每个样本，
        # 然后将矩阵中的m行通过加总压缩为一行，并且除以样本数量，即得到权重向量的每一个元素，
        # 它们合起来就是梯度的方向
        gradient_w = 1. / N * np.sum((z-y) * x, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = 1. / N * np.sum(z-y)
        return gradient_w, gradient_b

    def update(self, gradient_w, gradient_b, eta=0.01):
        """
        更新权重
        :param gradient_w:
        :param gradient_b:
        :param eta:
        :return:
        """
        self.w -= eta * gradient_w
        self.b -= eta * gradient_b

    def train(self, training_data, num_epoches, batch_size=10, eta=0.01):
        """
        训练
        :param training_data: 训练数据
        :param num_epoches: 轮次
        :param batch_size: 每一批的样本数量
        :param eta:
        :return:
        """
        n = len(training_data)
        losses = []
        for epoch_id in range(num_epoches):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
            for iter_id, mini_batch in enumerate(mini_batches):
                x = mini_batch[:, :-1]
                y = mini_batch[:, -1:]
                a = self.forward(x)
                loss = self.loss(a, y)
                gradient_w, gradient_b = self.gradient(x, y)
                self.update(gradient_w, gradient_b, eta)
                losses.append(loss)
                print('Epoch {:3d} / iter {:3d}, loss = {:.4f}'.format(epoch_id, iter_id, loss))
        return losses