import numpy as np
import cvxpy as cp
from typing import List, Dict, Any
import Task


class Client:
    def __init__(self, step_num: int, objectives: List[callable], data: List[np.ndarray]):
        """
        客户端类，包含本地目标和数据
        :param objectives: 客户端的多目标函数列表
        :param step_num: 每一个任务的步长
        :param data: 本地数据集（每个目标对应一个数据子集）
        """
        self.objectives = objectives
        self.step_num = step_num
        self.data = data

    def compute_gradients(self, x, stochastic: bool = False, batch_size: int = 16) -> dict:
        """
        计算本地梯度（全梯度或随机梯度），并储存为字典形式
        :param x: 当前每一个任务模型参数
        :param stochastic: 是否使用随机梯度
        :param batch_size: 随机梯度的批量大小
        :return: 各目标函数的梯度列表
        """
        gradients = []
        # 将每一个任务储存为字典形式
        test_gradients_dict = {}
        for i, obj in enumerate(self.objectives):
            test_gradients = []
            for j in range(self.step_num):
                #按任务计算梯度
                if stochastic:
                    # 随机采样批次数据
                    idx = np.random.choice(len(self.data[i]), batch_size, replace=False)
                    task = Task(obj, self.data[i][idx])
                    grad = task.test_compute_gradients()
                else:
                    # 全梯度计算
                    task = Task(obj, self.data[i])
                    grad = task.test_compute_gradients()
                test_gradients.append(grad) #单任务所有轮次的梯度
            test_gradients_dict[i] = test_gradients

        return test_gradients_dict