from Client import client_local_train
from ClientModel import ClientMTLModel
import torch
import torch.nn as nn
from data_load import CompositeDatasetGenerator, CompositeDataset, split_data_to_servers
from torch.utils.data import Dataset, DataLoader, Subset
from Service import ServerSharedModel
from Graient import federated_aggregation
import matplotlib.pyplot as plt

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
#生成测试数据
batch_images_test, batch_labels_test = generator.generate_batch(600)
full_dataset_test = CompositeDataset(batch_images, batch_labels)
train_loader_test = DataLoader(full_dataset_test, batch_size=64, shuffle=True)

# 初始化服务端模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
server_model = ServerSharedModel().to(device)
client_model = ClientMTLModel(server_model).to(device)
criterion = nn.CrossEntropyLoss()
global_learn_rate = 0.1


# 模拟5个客户端
num_clients = 5

# 联邦训练轮数
num_rounds = 100

for round in range(num_rounds):
    print(f"=== Federal Round {round + 1}/{num_rounds} ===")
    #统计量
    total_loss_task1 = 0
    total_loss_task2 = 0
    total_correct_task1 = 0
    total_correct_task2 = 0


    # 客户端本地训练
    client_models = []
    client_models_gard = {}
    for client_idx, dataset in client_datasets.items():
        # 加载本地数据
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
        # 本地多任务训练
        client_model, client_gard = client_local_train(client_model, server_model.feature_extractor.state_dict(), train_loader, num_epochs=5)
        client_models.append(client_model)
        client_models_gard[client_idx] = client_gard

    # 服务端聚合共享层参数
    server_model = federated_aggregation(server_model, client_models_gard, global_learn_rate)

    # 评估全局模型（以客户端0为例）
    client0_model = client_models[0].to(device)
    client0_model.eval()
    loss_history = {'task1': [], 'task2': []}


    with torch.no_grad():
        for data, (target_task1, target_task2) in train_loader_test:
            data = data.to(device)
            pred_task1, pred_task2 = client0_model(data)

            # loss
            total_loss_task1 += criterion(pred_task1, target_task1)
            loss_history['task1'].append(total_loss_task1)

            total_loss_task2 += criterion(pred_task2, target_task2)
            loss_history['task2'].append(total_loss_task2)

            # correct
            pred1 = pred_task1.argmax(dim=1, keepdim=True)
            total_correct_task1 += pred1.eq(target_task1.view_as(pred1)).sum().item()

            pred2 = pred_task2.argmax(dim=1, keepdim=True)
            total_correct_task2 += pred2.eq(target_task2.view_as(pred2)).sum().item()

    total_loss_task1 /= len(train_loader_test.dataset)
    total_loss_task2 /= len(train_loader_test.dataset)

    print(
        "Client 0 Test - task1 loss:{}".format(total_loss_task1), "task2 loss:{}".format(total_loss_task2))
    print(
        "Client 0 Test - task1 correct:{}".format(total_correct_task1 / len(train_loader_test.dataset)), "task2 correct:{}".format(total_correct_task2 / len(train_loader_test.dataset)))
    print("----------------------------------------------")

# 绘制损失曲线
plt.figure(figsize=(10, 6))
for cid in ['task1', 'task2']:
    plt.plot(loss_history[cid],
             label=f'{cid} (Final: {loss_history[cid][-1]:.4f})')

plt.title("Federated Multi-Task Learning Loss")
plt.xlabel("Global Epoch")
plt.ylabel("Average Local Loss")
plt.legend()
plt.grid(True)

# 保存图像
plt.savefig('fmtl_loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()
