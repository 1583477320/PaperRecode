import numpy as np
import torch
from collections import defaultdict
from scipy.optimize import minimize_scalar

'''
对每一个任务的梯度加权求和
'''


def aggregate_clients(client_dict):
    def average_across_clients(input_dict):
        # 初始化中间存储结构：share -> task -> [所有client的tensor]
        intermediate = defaultdict(lambda: defaultdict(list))

        # 遍历所有客户端
        for client_data in input_dict.values():
            # 遍历每个任务
            for task_name, task_data in client_data.items():
                # 遍历每个共享部分
                for share_name, tensor in task_data.items():
                    intermediate[share_name][task_name].append(tensor)

        # 计算平均值并构建结果
        result = {}
        for share_name, tasks in intermediate.items():
            share_dict = {}
            for task_name, tensors in tasks.items():
                # 堆叠张量并计算平均值
                stacked = torch.stack(tensors)
                avg_tensor = torch.mean(stacked, dim=0)
                share_dict[task_name] = avg_tensor
            result[share_name] = share_dict

        return result

    average_tensors = average_across_clients(client_dict)

    # 求凸优化
    lambda_1, lambda_2 = optimize_tensors(average_tensors['feature_extractor.1.weight']['task1'],
                                          average_tensors['feature_extractor.1.weight']['task2'])
    lambda_3, lambda_4 = optimize_tensors(average_tensors['feature_extractor.1.bias']['task1'],
                                          average_tensors['feature_extractor.1.bias']['task2'])

    return [lambda_1 * average_tensors['feature_extractor.1.weight']['task1'] + lambda_2 * average_tensors['feature_extractor.1.weight']['task2'],
            lambda_3 * average_tensors['feature_extractor.1.bias']['task1'] + lambda_4 * average_tensors['feature_extractor.1.bias']['task2']]


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