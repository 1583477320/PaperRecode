import numpy as np
import torch
from collections import defaultdict


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
    return [average_tensors["task1"][0] + average_tensors["task2"][0], average_tensors["task1"][1] + average_tensors["task2"][1]]


def weighted_sum_parameters(aggregated_result, task_weights=(0.5, 0.5)):
    """
    对聚合后的参数进行加权求和
    参数：
        aggregated_result: aggregate_clients函数的返回结果
        task_weights: 元组格式(task1权重, task2权重)

    返回：
        {"shared_layers.0.weight": 加权和, "shared_layers.0.bias": 加权和}
    """

    # 提取各任务参数
    def extract_param(task_name, param_name):
        for param_dict in aggregated_result[task_name]:
            if param_name in param_dict:
                return param_dict[param_name]
        raise ValueError(f"参数 {param_name} 在 {task_name} 中不存在")

    # 提取weight参数
    task1_w = extract_param("task1_head", "shared_layers.0.weight")
    task2_w = extract_param("task2_head", "shared_layers.0.weight")

    # 提取bias参数
    task1_b = extract_param("task1_head", "shared_layers.0.bias")
    task2_b = extract_param("task2_head", "shared_layers.0.bias")

    # 计算加权和（自动广播处理不同形状）
    weight_sum = task_weights[0] * task1_w + task_weights[1] * task2_w
    bias_sum = task_weights[0] * task1_b + task_weights[1] * task2_b

    return {
        "shared_layers.0.weight": weight_sum,
        "shared_layers.0.bias": bias_sum
    }
