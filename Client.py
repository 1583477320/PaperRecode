import numpy as np
from ClientModel import ClientMTLModel
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict


# -----------------客户端训练逻辑---------------
def client_local_train(client_model, train_loader, tasks=["task1", "task2"], num_epochs=5):
    """客户端本地多任务训练"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client_model.train()
    optimizer = optim.SGD(client_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # 初始化梯度累积器：按任务存储共享层梯度
    grad_accumulator = {
        task_id: [torch.zeros_like(p) for p in client_model.feature_extractor.parameters()]
        for task_id in tasks
    }
    for _ in range(num_epochs):
        for data, (target_task1, target_task2) in train_loader:
            data, target_task1, target_task2 = data.to(device), target_task1.to(device), target_task2.to(
                device)
            optimizer.zero_grad()
            # total_grad = [torch.zeros_like(param) for param in client_model.feature_extractor.parameters()]
            # total_loss = 0

            # 分别计算每个任务的梯度并累积
            output1 = client_model(data)[0]
            output2 = client_model(data)[0]

            # 计算损失
            loss1 = criterion(output1, target_task1)
            loss2 = criterion(output2, target_task2)
            total_loss = loss1 + loss2

            # 单次反向传播（自动累加多任务梯度）
            total_loss.backward()  # loss1和loss2的梯度自动叠加

            # 手动累积梯度（如果需要记录各任务梯度）
            with torch.no_grad():  # 避免干扰自动梯度计算
                grad_accumulator["task1"][0] += client_model.feature_extractor[1].weight.grad.clone()
                grad_accumulator["task1"][1] += client_model.feature_extractor[1].bias.grad.clone()
                grad_accumulator["task2"][0] += client_model.feature_extractor[1].weight.grad.clone()
                grad_accumulator["task2"][1] += client_model.feature_extractor[1].bias.grad.clone()

            # 更新参数（仅一次）
            optimizer.step()

    # 计算每个任务的平均梯度
    avg_gradients = {}
    for task_id in tasks:
        avg_gradients[task_id] = [
            grad / num_epochs for grad in grad_accumulator[task_id]
        ]
    return client_model, avg_gradients
