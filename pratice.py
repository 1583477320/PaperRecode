import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
import pandas as pd
import math
import sklearn.preprocessing as sk
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
import random
from Service import ServerSharedModel
from torch.utils.data import DataLoader
from ClientModel import ClientMTLModel
from data_load import CompositeDatasetGenerator, CompositeDataset, split_data_to_servers
import copy
import torch.nn.functional as F

seed = 42
random.seed(seed)
torch.cuda.manual_seed_all(seed)

N = 10000
M = 100
c = 0.5
p = 0.9
k = np.random.randn(M)
u1 = np.random.randn(M)
u1 -= u1.dot(k) * k / np.linalg.norm(k)**2
u1 /= np.linalg.norm(u1)
k /= np.linalg.norm(k)
u2 = k
w1 = c*u1
w2 = c*(p*u1+np.sqrt((1-p**2))*u2)
X = np.random.normal(0, 1, (N, M))
eps1 = np.random.normal(0, 0.01)
eps2 = np.random.normal(0, 0.01)
Y1 = np.matmul(X, w1) + np.sin(np.matmul(X, w1))+eps1
Y2 = np.matmul(X, w2) + np.sin(np.matmul(X, w2))+eps2
split = list(np.random.permutation(N))

# 准备原始数据集
# 合成数据
generator = CompositeDatasetGenerator(
    r"./OpenDataLab___MultiMNIST/raw/multi-mnist/data/train-images-idx3-ubyte",
    r"./OpenDataLab___MultiMNIST/raw/multi-mnist/data/train-labels-idx1-ubyte"
)

# 生成一个批次
batch_images, batch_labels = generator.generate_batch(6000)
full_dataset = CompositeDataset(batch_images, batch_labels)  # 使用前文生成的数据

#生成测试数据
batch_images_test, batch_labels_test = generator.generate_batch(256)
full_dataset_test = CompositeDataset(batch_images, batch_labels)


input_size, feature_size = 36*36, 36*36
shared_layer_size = 64
tower_h1 = 32
tower_h2 = 10
output_size = 1
LR = 0.01
epoch = 100
mb_size = 100
cost1tr = []
cost2tr = []
cost1D = []
cost2D = []
cost1ts = []
cost2ts = []
costtr = []
costD = []
costts = []

# class MTLnet(nn.Module):
#     def __init__(self):
#         super(MTLnet, self).__init__()
#
#         self.sharedlayer = nn.Sequential(
#             nn.Linear(feature_size, shared_layer_size),
#             nn.ReLU(),
#             nn.Dropout()
#         )
#         self.tower1 = nn.Sequential(
#             nn.Linear(shared_layer_size, tower_h1),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Linear(tower_h1, tower_h2)
#         )
#         self.tower2 = nn.Sequential(
#             nn.Linear(shared_layer_size, tower_h1),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Linear(tower_h1, tower_h2)
#         )

class ClientMTLModel(nn.Module):
    def __init__(self, server_model):
        super().__init__()
        # 继承服务端的共享特征层
        self.feature_extractor = copy.deepcopy(server_model.feature_extractor)
        # 本地任务头
        self.task1_head = nn.Sequential(
            nn.Linear(256, tower_h1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(tower_h1, tower_h2)
        )  # 任务1
        self.task2_head = nn.Sequential(
            nn.Linear(256, tower_h1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(tower_h1, tower_h2)
        )  # 任务2

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.task1_head(features), self.task2_head(features)

class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.task1_head = nn.Linear(50, tower_h2)
        self.task2_head = nn.Linear(50, tower_h2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return self.task1_head(x), self.task2_head(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# server_model = ServerSharedModel().to(device)
# MTL = ClientMTLModel(server_model)
MTL = CNNMnist()
optimizer = torch.optim.Adam(MTL.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

#loss函数记录
loss_history = {'task1': {"batch_size 16":[], "batch_size 64":[],"batch_size 128":[], "batch_size 256":[]},
                'task2': {"batch_size 16":[], "batch_size 64":[],"batch_size 128":[], "batch_size 256":[]}}

MTL.train()
batch_size_list = [64,128,256]

for batch_size in batch_size_list:
    print('--------batch_size{}-----------------'.format(batch_size))
    for it in range(epoch):
        epoch_cost = []
        epoch_cost1 = []
        epoch_cost2 = []
        loss1D = 0
        loss2D = 0
        # num_minibatches = int(input_size / mb_size)
        # minibatches = random_mini_batches(X_train, Y1_train, Y2_train, mb_size)
        minibatches = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
        for minibatch in minibatches:
            XE,(YE1, YE2) = minibatch

            Yhat1, Yhat2 = MTL(XE)
            l1 = loss_func(Yhat1, YE1)
            l2 = loss_func(Yhat2, YE2)
            loss = (l1 + l2) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_cost.append(loss)
            epoch_cost1.append(l1)
            epoch_cost2.append(l2)
        # costtr.append(torch.mean(torch.tensor(epoch_cost)))
        # cost1tr.append(torch.mean(torch.tensor(epoch_cost1)))
        # cost2tr.append(torch.mean(torch.tensor(epoch_cost2)))

        loss_history['task1']["batch_size {}".format(batch_size)].append(torch.mean(torch.tensor(epoch_cost1)))
        loss_history['task2']["batch_size {}".format(batch_size)].append(torch.mean(torch.tensor(epoch_cost2)))
        with torch.no_grad():
            data_test = DataLoader(full_dataset_test, batch_size=batch_size,shuffle=True)
            for X_valid,(Y1_valid, Y2_valid) in data_test:
                Yhat1D, Yhat2D = MTL(X_valid)
                l1D = loss_func(Yhat1D, Y1_valid)
                l2D = loss_func(Yhat2D, Y2_valid)
                loss1D += l1D
                loss2D += l2D
            cost1D.append(torch.mean(loss1D))
            cost2D.append(torch.mean(loss2D))
            costD.append((torch.mean(loss1D) + torch.mean(loss2D)) / 2)
            print('Iter-{}-{}; Total loss: {:.4}'.format(batch_size, it, loss))

    # plt.figure(figsize=(10, 6))
    # plt.plot(costtr, '-r'
    #          # ,costD, '-b'
    #          )
    # plt.ylabel('total cost')
    # plt.xlabel('iterations (per tens)')
    # plt.show()
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(cost1tr, '-r'
    #          # , cost1D, '-b'
    #          )
    # plt.ylabel('task 1 cost')
    # plt.xlabel('iterations (per tens)')
    # plt.show()
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(cost2tr, '-r'
    #          # , cost2D, '-b'
    #          )
    # plt.ylabel('task 2 cost')
    # plt.xlabel('iterations (per tens)')
    # plt.show()

# 绘制损失曲线
plt.figure(figsize=(10, 6))
task1_loss = loss_history["task1"]
for i in batch_size_list:
    plt.plot(task1_loss["batch_size {}".format(i)], label="batch_size {}".format(i))
plt.title("Task1 Loss")
plt.xlabel("Global Epoch")
plt.ylabel("Average Local Loss")
plt.legend()
plt.grid(False)
# 保存图像
plt.savefig('task1_loss_curve.png', dpi=300, bbox_inches='tight')


plt.figure(figsize=(10, 6))
task2_loss = loss_history["task2"]
for i in batch_size_list:
    plt.plot(task2_loss["batch_size {}".format(i)], label="batch_size {}".format(i))
plt.title("Task2 Loss")
plt.xlabel("Global Epoch")
plt.ylabel("Average Local Loss")
plt.legend()
plt.grid(False)

# 保存图像
plt.savefig('task2_loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()