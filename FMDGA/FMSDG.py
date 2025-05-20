from Client import client_local_train
from ClientModel import ClientMTLModel
import torch
import torch.nn as nn
from data_load import generate_multi_mnist, split_data_to_servers, DuplicatedLabelMNIST,OddLabelMNIST,generate_multi_mnist_non_iid_clients
from torch.utils.data import Dataset, DataLoader, Subset
from Service import ServerSharedModel
from Graient import federated_aggregation
import matplotlib.pyplot as plt
import numpy as np
import random


# 超参设置
num_servers = 10  # 模拟客户端个数
num_rounds = 100  # 通讯轮数
batch_size_list = [256]  # 训练batch_size列表
global_learn_rate = 0.1  # 服务端学习率
num_epochs = 5  # 客户端训练轮次
local_rate = 0.1  # 客户端学习率


# # 准备原始数据集
# # 不同分类生成一个批次
# train_dataset = generate_multi_mnist(num_samples=60000)
#
# #生成测试数据
# test_dataset = generate_multi_mnist(num_samples=6000, train=False)


# #重复标签生成一个批次
# train_dataset = OddLabelMNIST(root='./data_Mnist', train=True)
#
# # 训练流程
# # ------------------------------
# #生成测试数据
# test_dataset = OddLabelMNIST(root='./data_Mnist', train=False)
#
#
#loss函数记录
loss_history = {'task1': {"batch_size 16":[], "batch_size 64":[],"batch_size 128":[], "batch_size 256":[]},
                'task2': {"batch_size 16":[], "batch_size 64":[],"batch_size 128":[], "batch_size 256":[]}}
#
# sample_index = [i for i in range(6000)] #假设取前500个训练数据
# X_train = []
# y_train = []
# for i in sample_index:
#     X = train_dataset[i][0]
#     X_train.append(X)
#     y = train_dataset[i][1]
#     y_train.append(y)
#
# sampled_train_data = [(X, y) for X, y in zip(X_train, y_train)] #包装为数据对
# # trainDataLoader = torch.utils.data.DataLoader(sampled_train_data, batch_size=256, shuffle=True)
#
# client_datasets = split_data_to_servers(sampled_train_data, num_servers=num_servers)
#
# sample_test_index = [i for i in range(256)] #假设取前500个训练数据
# X_test = []
# y_test = []
# for i in sample_test_index:
#     X = test_dataset[i][0]
#     X_train.append(X)
#     y = test_dataset[i][1]
#     y_train.append(y)
#
# sampled_test_data = [(X, y) for X, y in zip(X_train, y_train)] #包装为数据对

train_data = generate_multi_mnist_non_iid_clients(num_clients=10)

test_data = generate_multi_mnist_non_iid_clients(train=False)

# 定义初始化函数
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

# 设置所有随机种子
def set_seed(seed):
    torch.manual_seed(seed)       # CPU随机种子
    torch.cuda.manual_seed(seed)  # GPU随机种子
    torch.cuda.manual_seed_all(seed)  # 多GPU时
    np.random.seed(seed)          # NumPy随机种子
    random.seed(seed)             # Python随机种子
    torch.backends.cudnn.deterministic = True  # 确保CUDA卷积算法确定
    torch.backends.cudnn.benchmark = False     # 关闭自动优化

for batch_size in batch_size_list:
    # 初始化服务端模型
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server_model = ServerSharedModel().to(device)
    server_model.apply(init_weights)  # 参数固定初始化
    client_model = ClientMTLModel(server_model).to(device)
    client_model.apply(init_weights)
    criterion = nn.CrossEntropyLoss()
    global_learn_rate = global_learn_rate
    client_models = []

    # 测试统计量
    total_loss_task1 = 0
    total_loss_task2 = 0
    total_correct_task1 = 0
    total_correct_task2 = 0
    # 选取对应测试数据
    train_loader_test = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    for round in range(num_rounds):
        print(f"======== batch_size {batch_size} ========")
        print(f"=== Federal Round {round + 1}/{num_rounds} ===")

        task1_loss_locals = []
        task2_loss_locals = []

        # 客户端本地训练
        client_models_gard = {}
        for client_idx, dataset in train_data.items():
            # 加载本地数据
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            # 本地多任务训练
            client_model, client_gard, task_loss = client_local_train(client_model, server_model.feature_extractor.state_dict(), train_loader, num_epochs=num_epochs, local_rate=local_rate)
            client_models.append(client_model)
            client_models_gard[client_idx] = client_gard
            #记录客户端各任务loss
            task1_loss_locals.append(task_loss[0])
            task2_loss_locals.append(task_loss[1])

        task1_loss_avg = sum(task1_loss_locals) / len(task1_loss_locals)
        task2_loss_avg = sum(task2_loss_locals) / len(task2_loss_locals)
        loss_history['task1']["batch_size {}".format(batch_size)].append(task1_loss_avg)
        loss_history['task2']["batch_size {}".format(batch_size)].append(task2_loss_avg)
        # 服务端聚合共享层参数
        server_model = federated_aggregation(server_model, client_models_gard, global_learn_rate=global_learn_rate)
        print(
            "task1 loss:{}".format(task1_loss_avg), "task2 loss:{}".format(task2_loss_avg))
        print("----------------------------------------------")

    # 评估全局模型（以客户端0为例）
    client0_model = client_models[0].to(device)
    client0_model.eval()


    with torch.no_grad():
        for data, (target1,target2) in train_loader_test:
            data = data.to(device)
            data, target_task1, target_task2 = data.to(device), target1.to(device), target2.to(
                device)

            pred_task1, pred_task2 = client0_model(data)

            # loss
            total_loss_task1 += criterion(pred_task1, target_task1)

            total_loss_task2 += criterion(pred_task2, target_task2)

            # correct
            pred1 = pred_task1.argmax(dim=1, keepdim=True)
            total_correct_task1 += pred1.eq(target_task1.view_as(pred1)).sum().item()

            pred2 = pred_task2.argmax(dim=1, keepdim=True)
            total_correct_task2 += pred2.eq(target_task2.view_as(pred2)).sum().item()

    # total_loss_task1 /= len(train_loader_test.dataset)
    # total_loss_task1 *= batch_size / 16
    # loss_history['task1']["batch_size {}".format(batch_size)].append(total_loss_task1)
    #
    # total_loss_task2 /= len(train_loader_test.dataset)
    # total_loss_task2 *= batch_size / 16
    # loss_history['task2']["batch_size {}".format(batch_size)].append(total_loss_task2)

    # print(
    #     "Client 0 Test - task1 loss:{}".format(total_loss_task1), "task2 loss:{}".format(total_loss_task2))
    print(
        "Client 0 Test - task1 correct:{}".format(total_correct_task1 / len(train_loader_test.dataset)), "task2 correct:{}".format(total_correct_task2 / len(train_loader_test.dataset)))


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
plt.savefig('task1_mulit_loss_curve.png', dpi=300, bbox_inches='tight')


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
plt.savefig('task2_mulit_loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()
