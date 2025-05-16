from Client import client_local_train
from ClientModel import ClientMTLModel
import torch
import torch.nn as nn
from data_load import generate_multi_mnist, CompositeDataset, split_data_to_servers,DuplicatedLabelMNIST
from torch.utils.data import Dataset, DataLoader, Subset
from Service import ServerSharedModel
from Graient import federated_aggregation
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
import torch.optim as optim

# 超参设置
num_servers = 10  # 模拟客户端个数
num_rounds = 100  # 通讯轮数
batch_size_list = [256]  # 训练batch_size列表
global_learn_rate = 0.1  # 服务端学习率
num_epochs = 100  # 客户端训练轮次
local_rate = 0.1  # 客户端学习率


# 准备原始数据集
# 合成数据

# 生成一个批次
train_dataset = DuplicatedLabelMNIST(root='./data_Mnist', train=True)

# 训练流程
# ------------------------------
#生成测试数据
test_dataset = DuplicatedLabelMNIST(root='./data_Mnist', train=False)

# # 生成一个批次
# batch_images, batch_labels = generate_multi_mnist(root='./data_Mnist',output_dir="./multi_mnist_data")
# train_dataset = CompositeDataset(batch_images, batch_labels)  # 使用前文生成的数据
#
# #生成测试数据
# batch_images_test, batch_labels_test = generate_multi_mnist(root='./data_Mnist',train=False,output_dir="./multi_mnist_data")
# test_dataset = CompositeDataset(batch_images, batch_labels)

sample_index = [i for i in range(6000)] #假设取前500个训练数据
X_train = []
y_train = []
for i in sample_index:
    X = train_dataset[i][0]
    X_train.append(X)
    y = train_dataset[i][1]
    y_train.append(y)

sampled_train_data = [(X, y) for X, y in zip(X_train, y_train)] #包装为数据对
# trainDataLoader = torch.utils.data.DataLoader(sampled_train_data, batch_size=256, shuffle=True)


sample_test_index = [i for i in range(256)] #假设取前500个训练数据
X_test = []
y_test = []
for i in sample_test_index:
    X = test_dataset[i][0]
    X_train.append(X)
    y = test_dataset[i][1]
    y_train.append(y)

sampled_test_data = [(X, y) for X, y in zip(X_train, y_train)] #包装为数据对


#loss函数记录
loss_history = {'task1': {"batch_size 16":[], "batch_size 64":[],"batch_size 128":[], "batch_size 256":[]},
                'task2': {"batch_size 16":[], "batch_size 64":[],"batch_size 128":[], "batch_size 256":[]}}

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

    # 测试统计量
    total_loss_task1 = 0
    total_loss_task2 = 0
    total_correct_task1 = 0
    total_correct_task2 = 0
    # 选取对应测试数据
    train_loader_test = DataLoader(sampled_test_data, batch_size=batch_size, shuffle=True
                                   )

    # # loss起始
    # init_loss_task1 = 0
    # init_loss_task2 = 0
    # with torch.no_grad():
    #     for data, (target_task1, target_task2) in train_loader_test:
    #         data = data.to(device)
    #         pred_task1, pred_task2 = client_model(data)
    #
    #         # loss
    #         init_loss_task1 += criterion(pred_task1, target_task1)
    #         init_loss_task2 += criterion(pred_task2, target_task2)
    # init_loss_task1 /= len(train_loader_test.dataset)
    # init_loss_task1 *= batch_size / 16
    # loss_history['task1']["batch_size {}".format(batch_size)].append(init_loss_task1)
    #
    # init_loss_task2 /= len(train_loader_test.dataset)
    # init_loss_task2 *= batch_size / 16
    # loss_history['task2']["batch_size {}".format(batch_size)].append(init_loss_task1)
    client_model.train()

    task1_loss = []
    task2_loss = []

    # 初始化梯度累积器：按任务存储共享层梯度
    grad_accumulator = {
        task_id: {name: [] for name, param in client_model.named_parameters() if "feature_extractor" in name}
        for task_id in ["task1", "task2"]
    }
    for round in range(num_rounds):
        print(f"======== batch_size {batch_size} ========")
        print(f"=== Federal Round {round + 1}/{num_rounds} ===")

        # 加载本地数据
        train_loader = DataLoader(sampled_train_data, batch_size=batch_size, shuffle=True)
        # 本地多任务训练
        # 客户端本地多任务训练
        optimizer = optim.SGD(client_model.parameters(), lr=local_rate)

        # 记录损失
        task1_batch_loss = []
        task2_batch_loss = []
        for data, target in train_loader:
            data, target_task1, target_task2 = data.to(device), target.T[0].to(device), target.T[1].to(
                device)
            # total_grad = [torch.zeros_like(param) for param in client_model.feature_extractor.parameters()]
            # total_loss = 0

            # 分别计算每个任务的梯度并累积
            output1 = client_model(data)[0]
            output2 = client_model(data)[1]

            # 计算损失
            loss1 = criterion(output1, target_task1)
            loss2 = criterion(output2, target_task2)
            total_loss = (loss1 + loss2)/2

            '''
            手动累积梯度（如果需要记录各任务梯度）,分别查看每个任务对共享层的梯度,retain_graph=True：在反向传播时，设置 retain_graph=True 以保留计算图，否则计算图会在第一次反向传播后被释放，无法进行第二次反向传播。
            梯度的累加：如果你不冻结共享层的梯度，梯度会自动累加。因此，在分别计算每个任务的梯度时，需要先清除共享层的梯度。
            优化器的 zero_grad() 方法：在每次反向传播之前，建议使用 optimizer.zero_grad() 或手动清除梯度，以避免梯度累积。
            '''

            # 任务1的梯度
            optimizer.zero_grad()  # 清除共享层的梯度
            loss1.backward(retain_graph=True)

            grads_task1 = {}
            for name, param in client_model.named_parameters():
                if param.grad is not None:
                    grads_task1[name] = param.grad.clone()

            # 任务2的梯度
            optimizer.zero_grad()  # 清除共享层的梯度
            loss2.backward()
            grads_task2 = {}
            for name, param in client_model.named_parameters():
                if param.grad is not None:
                    grads_task2[name] = param.grad.clone()

            optimizer.zero_grad()
            auto_up = False
            if auto_up:
                # # 单次反向传播（自动累加多任务梯度）
                total_loss.backward()  # loss1和loss2的梯度自动叠加
            else:
                # 手动合并梯度：共享层取均值，任务层保留各自的梯度
                for name, param in client_model.named_parameters():
                    if "feature_extractor" in name:  # 共享层
                        if name in grads_task1 and name in grads_task2:
                            # 共享层的梯度取均值
                            param.grad = (grads_task1[name] + grads_task2[name]) / 2
                    else:  # 任务层
                        # 任务层的梯度保留各自的梯度
                        if name in grads_task1:
                            param.grad = grads_task1[name]
                        elif name in grads_task2:
                            param.grad = grads_task2[name]


            #梯度记录
            for name, param in client_model.named_parameters():
                if "feature_extractor" in name:  # 共享层
                    if name in grads_task1 :
                        # 共享层的梯度取均值
                        grad_accumulator['task1']["{}".format(name)].append(param)

            # 更新参数
            optimizer.step()

            # 记录损失
            task1_batch_loss.append(loss1.item())
            task2_batch_loss.append(loss2.item())

        # 记录客户端各任务loss
        loss_history['task1']["batch_size {}".format(batch_size)].append(sum(task1_batch_loss) / len(task1_batch_loss))
        loss_history['task2']["batch_size {}".format(batch_size)].append(sum(task2_batch_loss) / len(task2_batch_loss))
        print(
            "task1 loss:{}".format(sum(task1_batch_loss) / len(task1_batch_loss)),
            "task2 loss:{}".format(sum(task2_batch_loss) / len(task2_batch_loss)))
        print("----------------------------------------------")
    # 初始化平均结果字典
    avg_grad = {}

    # 遍历每个name
    for name in grad_accumulator['task1'].keys():
        # 收集所有任务中该name的张量
        tensors = [grad_accumulator[task][name] for task in grad_accumulator.keys()]
        # 计算平均
        avg = sum(tensors[0]) / len(tensors[0])
        # 存储到avg_grad字典中
        if name not in avg_grad:
            avg_grad[name] = {}
        avg_grad[name] = avg
    # 服务端聚合共享层参数
    # server_model = federated_aggregation(server_model, client_models_gard, global_learn_rate=global_learn_rate)
    # 评估全局模型（以客户端0为例）
    client_model.eval()

    with torch.no_grad():
        test_loader = DataLoader(sampled_test_data, batch_size=batch_size, shuffle=True)
        for data, target in test_loader:
            data = data.to(device)
            pred_task1, pred_task2 = client_model(data)

            # loss
            total_loss_task1 += criterion(pred_task1, target.T[0])
            total_loss_task2 += criterion(pred_task2, target.T[1])

            # correct
            pred1 = pred_task1.argmax(dim=1, keepdim=True)
            total_correct_task1 += pred1.eq(target.T[0].view_as(pred1)).sum().item()

            pred2 = pred_task2.argmax(dim=1, keepdim=True)
            total_correct_task2 += pred2.eq(target.T[1].view_as(pred2)).sum().item()

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
plt.savefig('hand_task1_loss_curve.png', dpi=300, bbox_inches='tight')


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
plt.savefig('hand_task2_loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()
