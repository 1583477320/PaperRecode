import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cvxpy as cp
import copy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

# 1. 多任务MNIST数据集生成
class MultiMNIST(Dataset):
    def __init__(self, mnist_dataset, num_tasks=2):
        self.data = []
        self.targets = []
        
        for img, label in mnist_dataset:
            # 生成双数字图像
            img1 = self.overlay_digit(img, label, position=(0,0))   # 左上角
            img2 = self.overlay_digit(img, (label+1)%10, position=(14,14)) # 右下角
            self.data.extend([img1, img2])
            self.targets.extend([label, (label+1)%10])

    def overlay_digit(self, base_img, digit, position):
        canvas = torch.zeros((28,28))
        digit_img = torchvision.datasets.MNIST(
            root='./data', train=True, download=True
        ).data[digit*6000] / 255.0  # 取第一个样本
        
        x, y = position
        canvas[y:y+14, x:x+14] = base_img
        canvas[y+14:y+28, x+14:x+28] = digit_img
        return canvas.unsqueeze(0)  # 添加通道维度

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# 2. 联邦客户端划分（非IID）
def create_non_iid_clients(num_clients=10):
    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = MultiMNIST(
        torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    )
    
    # 按标签划分客户端
    client_datasets = [[] for _ in range(num_clients)]
    labels = np.array(full_dataset.targets)
    
    for client_id in range(num_clients):
        # 每个客户端随机分配两个主标签
        main_labels = np.random.choice(10, 2, replace=False)
        indices = np.where(np.isin(labels, main_labels))[0]
        selected = np.random.choice(indices, 1200, replace=False)  # 每个客户端120样本
        
        client_datasets[client_id] = torch.utils.data.Subset(full_dataset, selected)
    
    return client_datasets

# 3. 多任务模型定义
class MultiTaskModel(nn.Module):
    def __init__(self, num_tasks=2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(32*14*14, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            ) for _ in range(num_tasks)]
        )
    
    def forward(self, x, task_id):
        features = self.shared(x)
        return self.heads[task_id](features)

# 4. 联邦优化算法核心实现
class FMOLOptimizer:
    def __init__(self, global_model, num_tasks=2, lr_local=0.1, lr_global=0.1):
        self.global_model = global_model
        self.num_tasks = num_tasks
        self.lr_local = lr_local
        self.lr_global = lr_global
        
    def client_update(self, client_data, K=10, batch_size=64, full_grad=False):
        local_model = copy.deepcopy(self.global_model)
        local_model.train()
        
        # 获取本地数据加载器
        loader = DataLoader(client_data, batch_size=batch_size if not full_grad else len(client_data), 
                          shuffle=True)
        
        # 本地K次更新
        for _ in range(K):
            for images, labels in loader:
                task_ids = torch.randint(0, self.num_tasks, (images.size(0),))
                
                # 多任务梯度计算
                grads = []
                for task in range(self.num_tasks):
                    mask = (task_ids == task)
                    if mask.sum() == 0:
                        continue
                    
                    # 前向传播
                    outputs = local_model(images[mask], task)
                    loss = nn.CrossEntropyLoss()(outputs, labels[mask])
                    
                    # 反向传播
                    local_model.zero_grad()
                    loss.backward()
                    
                    # 收集梯度
                    task_grad = [param.grad.detach().clone() for param in local_model.parameters()]
                    grads.append(task_grad)
                
                # 参数更新（多梯度下降）
                if len(grads) > 0:
                    avg_grad = [sum(g) / len(grads) for g in zip(*grads)]
                    for param, g in zip(local_model.parameters(), avg_grad):
                        param.data -= self.lr_local * g
        
        # 计算梯度差异
        delta = []
        for global_param, local_param in zip(self.global_model.parameters(), local_model.parameters()):
            delta.append((global_param.data - local_param.data) / self.lr_local)
        
        return delta
    
    def server_aggregate(self, client_grads):
        # 解决二次规划问题
        S = self.num_tasks
        grad_matrix = [torch.cat([g.reshape(-1) for g in task_grads]) for task_grads in zip(*client_grads)]
        
        # 构建优化问题
        lambda_var = cp.Variable(S)
        objective = cp.Minimize(cp.sum_squares(sum([lambda_var[i]*grad_matrix[i] for i in range(S)])))
        constraints = [lambda_var >= 0, cp.sum(lambda_var) == 1]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        
        # 计算下降方向
        descent_dir = [sum([lambda_var.value[i]*grad_matrix[i][j] 
                        for i in range(S)])
                     for j in range(len(grad_matrix[0]))]
        
        return descent_dir
    
    def global_update(self, descent_dir):
        # 更新全局模型
        with torch.no_grad():
            for param, d in zip(self.global_model.parameters(), descent_dir):
                param.data -= self.lr_global * d.reshape(param.shape)

# 5. 训练循环与可视化
def train_fmol(config):
    # 初始化
    num_clients = 10
    num_rounds = 100
    client_datasets = create_non_iid_clients(num_clients)
    
    # 创建全局模型
    global_model = MultiTaskModel(num_tasks=2)
    optimizer = FMOLOptimizer(global_model, lr_local=0.1, lr_global=0.1)
    
    # 存储损失
    losses = []
    
    for round in range(num_rounds):
        # 客户端并行更新
        client_grads = []
        for client_data in client_datasets:
            delta = optimizer.client_update(
                client_data, 
                K=config['K'], 
                batch_size=config['batch_size'],
                full_grad=(config['batch_size'] == 256)
            )
            client_grads.append(delta)
        
        # 服务器聚合
        aggregated_grads = list(zip(*client_grads))  # 转置梯度结构
        descent_dir = optimizer.server_aggregate(aggregated_grads)
        optimizer.global_update(descent_dir)
        
        # 评估
        if round % 5 == 0:
            total_loss = 0
            for client_data in client_datasets:
                loader = DataLoader(client_data, batch_size=128)
                for images, labels in loader:
                    task_ids = torch.randint(0, 2, (images.size(0),))
                    outputs = global_model(images, task_ids)
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                    total_loss += loss.item()
            losses.append(total_loss / len(client_datasets))
    
    return losses

# 6. 实验配置与可视化
if __name__ == "__main__":
    experiments = [
        {'K': 1, 'batch_size': 16},
        {'K': 5, 'batch_size': 64},
        {'K': 10, 'batch_size': 128},
        {'K': 20, 'batch_size': 256}
    ]
    
    plt.figure(figsize=(10,6))
    for config in experiments:
        print(f"Running experiment K={config['K']}, batch={config['batch_size']}")
        losses = train_fmol(config)
        plt.plot(np.linspace(0, 100, len(losses)), losses, 
                 label=f"K={config['K']}, BS={config['batch_size']}")
    
    plt.xlabel('Communication Rounds')
    plt.ylabel('Average Loss')
    plt.title('Federated Multi-Objective Learning Convergence')
    plt.legend()
    plt.grid(True)
    plt.savefig('fmol_convergence.png')
    plt.show()