import torch
import torch.nn as nn
import copy

'''
客户端模型
'''


class ClientMTLModel(nn.Module):
    def __init__(self, server_model):
        super().__init__()
        # 继承服务端的共享特征层
        self.feature_extractor = copy.deepcopy(server_model.feature_extractor)
        # 本地任务头
        self.task1_head = nn.Linear(128, 10)  # 任务1
        self.task2_head = nn.Linear(128, 10)  # 任务2

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.task1_head(features), self.task2_head(features)
