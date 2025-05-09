import numpy as np
import torch
from collections import defaultdict
from scipy.optimize import minimize_scalar

'''
对每一个任务的梯度加权求和
'''


def aggregate_clients(client_dict):
    # 初始化存储结构
    grouped = defaultdict(lambda: defaultdict(list))

    # 收集所有客户端的张量
    for server_data in client_dict.values():
        for key in server_data:
            for pos_idx, tensor in enumerate(server_data[key]):
                grouped[key][pos_idx].append(tensor)

    # 计算平均值
    averaged = defaultdict(list)
    for key in sorted(grouped.keys()):
        for pos in sorted(grouped[key].keys()):
            tensors = grouped[key][pos]

            # 自动检测张量类型并计算平均
            if isinstance(tensors[0], torch.Tensor):
                stacked = torch.stack(tensors, dim=0)
                avg_tensor = torch.mean(stacked, dim=0)
            else:
                stacked = np.stack(tensors, axis=0)
                avg_tensor = np.mean(stacked, axis=0)

            averaged[key].append(avg_tensor)

    average_tensors = dict(averaged)

    # 求凸优化
    lambda_1, lambda_2 = optimize_tensors(average_tensors["task1"][0], average_tensors["task2"][0])
    lambda_3, lambda_4 = optimize_tensors(average_tensors["task1"][1], average_tensors["task2"][1])

    return [lambda_1 * average_tensors["task1"][0] + lambda_2 * average_tensors["task2"][0],
            lambda_3 * average_tensors["task1"][1] + lambda_4 * average_tensors["task2"][1]]


def optimize_tensors(tensor1, tensor2):
    """优化两个张量的线性组合系数，返回使范数最小的系数"""

    # 定义目标函数（闭包捕获外部张量）
    def objective(x1):
        x2 = 1 - x1
        combination = x1 * tensor1 + x2 * tensor2
        return torch.norm(combination) ** 2

    # 使用更适合单变量有界优化的Brent方法
    result = minimize_scalar(
        objective,
        bounds=(0, 1),
        method='bounded',
        options={'xatol': 1e-8}  # 设置更高的收敛精度
    )

    # 解析结果
    x1_opt = result.x
    x2_opt = 1 - x1_opt
    min_norm = result.fun

    return x1_opt, x2_opt