from Client import client_local_train
from ClientModel import ClientMTLModel
import torch
import torch.nn as nn
from data_load import CompositeDatasetGenerator, CompositeDataset, split_data_to_servers
from torch.utils.data import Dataset, DataLoader, Subset
from Service import ServerSharedModel
from Graient import federated_aggregation

# 准备原始数据集
# 合成数据
generator = CompositeDatasetGenerator(
    r"C:\Users\15834\PycharmProjects\paper_recode\OpenDataLab___MultiMNIST\raw\multi-mnist\data\train-images-idx3-ubyte",
    r"C:\Users\15834\PycharmProjects\paper_recode\OpenDataLab___MultiMNIST\raw\multi-mnist\data\train-labels-idx1-ubyte"
)

# 生成一个批次
batch_images, batch_labels = generator.generate_batch(6000)
full_dataset = CompositeDataset(batch_images, batch_labels)  # 使用前文生成的数据

# 分配数据到5个服务器
client_datasets = split_data_to_servers(full_dataset, num_servers=5)

# 训练流程
# ------------------------------
# 初始化服务端模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
server_model = ServerSharedModel().to(device)
criterion = nn.CrossEntropyLoss()
global_learn_rate = 0.003
total_loss_task1 = 0
total_loss_task2 = 0

# 模拟5个客户端
num_clients = 5

# 联邦训练轮数
num_rounds = 100

for round in range(num_rounds):
    print(f"=== Federal Round {round + 1}/{num_rounds} ===")

    # 客户端本地训练
    client_models = []
    client_models_gard = {}
    for client_idx, dataset in client_datasets.items():
        # 下载服务端共享层参数
        client_model = ClientMTLModel(server_model).to(device)
        # 加载本地数据
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
        # 本地多任务训练
        client_model, client_gard = client_local_train(client_model, train_loader)
        client_models.append(client_model)
        client_models_gard[client_idx] = client_gard

    # 服务端聚合共享层参数
    server_model = federated_aggregation(server_model, client_models_gard, global_learn_rate)

    # 评估全局模型（以客户端0为例）
    client0_model = client_models[0]
    test_loader = DataLoader(full_dataset, batch_size=64)
    client0_model.eval()
    correct_cls, total_cls, total_reg = 0, 0, 0.0

    with torch.no_grad():
        for data, (target_task1, target_task2) in test_loader:
            data = data.to(device)
            pred_task1, pred_task2 = client0_model(data)

            # # 分类任务准确率
            # _, predicted = torch.max(pred_cls.cpu().data, 1)
            # correct_cls += (predicted == target_cls).sum().item()
            # total_cls += target_cls.size(0)

            # loss
            total_loss_task1 += criterion(pred_task1, target_task1)
            total_loss_task2 += criterion(pred_task2, target_task2)

    print(
        "Client 0 Test - task1 loss:{}".format(total_loss_task1), "task2 loss:{}".format(total_loss_task2))
