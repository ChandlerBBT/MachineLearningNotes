# encoding:utf-8
# project: MachineLearningNotes
# author: chandler
# file: intro_example.py
# create_time: 2020/4/20
# IDE: Pycharm


"""入门小例子"""

import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0,100,100)
X = np.c_[X, np.ones(100)]
w = np.asarray([3,2])
Y = X.dot(w)
X = X.astype("float")
Y = Y.astype("float")
X[:,0] += np.random.normal(size=(X[:,0].shape))*3  # 给每个样本添加一些噪声

Y = Y.reshape(100,1)  # 此前,Y.shape = (100,)

# 初始化权重
w = np.random.random(size=(2,1))
# 跟新参数
epoches = 100
eta = 0.0000001
losses = []
for _ in range(epoches):
    dw = -2 * X.T.dot(Y-X.dot(w))
    w = w - eta * dw
    losses.append((Y-X.dot(w)).T.dot(Y-X.dot(w)).reshape(-1))


plt.scatter(X[:,0],Y)
# plt.plot(np.arange(0,100).reshape((100,1)), Y, "r")
plt.plot(X[:,0], X.dot(w),"r")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()