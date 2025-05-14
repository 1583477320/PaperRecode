from Client import client_local_train
from ClientModel import ClientMTLModel
import torch
import torch.nn as nn
from data_load import CompositeDatasetGenerator, CompositeDataset, split_data_to_servers
from torch.utils.data import Dataset, DataLoader, Subset
from Service import ServerSharedModel
from Graient import federated_aggregation
import matplotlib.pyplot as plt
import numpy as np
import random


# 超参设置
num_servers = 10  # 模拟客户端个数
num_rounds = 100  # 通讯轮数
batch_size_list = [16, 64, 128, 256]  # 训练batch_size列表
global_learn_rate = 0.1  # 服务端学习率
num_epochs = 10  # 客户端训练轮次
local_rate = 0.1  # 客户端学习率



# 准备原始数据集
# 合成数据
generator = CompositeDatasetGenerator(
    r".\OpenDataLab___MultiMNIST\raw\multi-mnist\data\train-images-idx3-ubyte",
    r".\OpenDataLab___MultiMNIST\raw\multi-mnist\data\train-labels-idx1-ubyte"
)

# 生成完整的训练数据
batch_images, batch_labels = generator.generate_batch(1000)
full_dataset = CompositeDataset(batch_images, batch_labels)  # 使用前文生成的数据

# 分配数据到5个服务器
client_datasets = split_data_to_servers(full_dataset, num_servers=num_servers)

# 训练流程
# ------------------------------
# 生成测试数据
batch_images_test, batch_labels_test = generator.generate_batch(256)
full_dataset_test = CompositeDataset(batch_images_test, batch_labels_test)

# loss函数记录
loss_history = {'task1': {"batch_size 16": [], "batch_size 64": [], "batch_size 128": [], "batch_size 256": []},
                'task2': {"batch_size 16": [], "batch_size 64": [], "batch_size 128": [], "batch_size 256": []}}

# 定义初始化函数
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

# 固定 DataLoader 的生成器
g = torch.Generator()
g.manual_seed(42)

def seed_worker(worker_id):
    worker_seed = 42
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
    print(f"======== batch_size {batch_size} ========")
    # 初始化服务端模型
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server_model = ServerSharedModel().to(device)
    server_model.apply(init_weights)  # 参数固定初始化
    client_model = ClientMTLModel(server_model).to(device)
    client_model.apply(init_weights)
    criterion = nn.CrossEntropyLoss()
    global_learn_rate = global_learn_rate

    # 选取对应测试数据
    train_loader_test = DataLoader(full_dataset_test, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker,
                                   generator=g)

    #loss起始
    init_loss_task1 = 0
    init_loss_task2 = 0
    with torch.no_grad():
        for data, (target_task1, target_task2) in train_loader_test:
            data = data.to(device)
            pred_task1, pred_task2 = client_model(data)

            # loss
            init_loss_task1 += criterion(pred_task1, target_task1)
            init_loss_task2 += criterion(pred_task2, target_task2)
    init_loss_task1 /= len(train_loader_test.dataset)
    init_loss_task1 *= batch_size / 16
    loss_history['task1']["batch_size {}".format(batch_size)].append(init_loss_task1)
    print("Client 0 Test - task1 loss init:{}".format(init_loss_task1), "task1 loss:{}".format(init_loss_task1))

    init_loss_task2 /= len(train_loader_test.dataset)
    init_loss_task2 *= batch_size / 16
    loss_history['task2']["batch_size {}".format(batch_size)].append(init_loss_task2)
    print("Client 0 Test - task2 loss init:{}".format(init_loss_task2), "task2 loss:{}".format(init_loss_task2))

    for round in range(1):
        print(f"=== Federal Round {round + 1}/{num_rounds} ===")
        # 统计量
        total_loss_task1 = 0
        total_loss_task2 = 0
        total_correct_task1 = 0
        total_correct_task2 = 0

        # 客户端本地训练
        client_models = []
        client_models_gard = {}
        for client_idx, dataset in client_datasets.items():
            # 加载本地数据
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,worker_init_fn=seed_worker,
        generator=g)
            # 本地多任务训练
            client_model, client_gard = client_local_train(client_model, server_model.feature_extractor.state_dict(),
                                                           train_loader, num_epochs=num_epochs, local_rate=local_rate)
            client_models.append(client_model)
            client_models_gard[client_idx] = client_gard

        # 服务端聚合共享层参数
        server_model = federated_aggregation(server_model, client_models_gard, global_learn_rate=global_learn_rate)

        # 评估全局模型（以客户端0为例）
        client0_model = client_models[0].to(device)
        client0_model.eval()

        with torch.no_grad():
            for data, (target_task1, target_task2) in train_loader_test:
                data = data.to(device)
                pred_task1, pred_task2 = client0_model(data)

                # loss
                total_loss_task1 += criterion(pred_task1, target_task1)

                total_loss_task2 += criterion(pred_task2, target_task2)

                # correct
                pred1 = pred_task1.argmax(dim=1, keepdim=True)
                total_correct_task1 += pred1.eq(target_task1.view_as(pred1)).sum().item()

                pred2 = pred_task2.argmax(dim=1, keepdim=True)
                total_correct_task2 += pred2.eq(target_task2.view_as(pred2)).sum().item()

        total_loss_task1 /= len(train_loader_test.dataset)
        total_loss_task1 *= batch_size/16
        loss_history['task1']["batch_size {}".format(batch_size)].append(total_loss_task1)

        total_loss_task2 /= len(train_loader_test.dataset)
        total_loss_task2 *= batch_size / 16
        loss_history['task2']["batch_size {}".format(batch_size)].append(total_loss_task2)

        print(
            "Client 0 Test - task1 loss:{}".format(total_loss_task1), "task2 loss:{}".format(total_loss_task2))
        print(
            "Client 0 Test - task1 correct:{}".format(total_correct_task1 / len(train_loader_test.dataset)),
            "task2 correct:{}".format(total_correct_task2 / len(train_loader_test.dataset)))
        print("----------------------------------------------")