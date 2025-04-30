import numpy as np
import cvxpy as cp
from typing import List, Dict, Any

class Task:
    def __init__(self, model, loss_function ):
        """
        计算每一个任务的梯度
        :param model: 机器学习模型
        :param loss_function:损失函数
        :param input_data: 输入数据
        :param target: 目标数据
        """
        self.model = model
        self.loss_function = loss_function

    def test_compute_gradients(self, init_params, input_data, target, b):
        model = self.model(init_params=init_params, input_data=input_data)
        output = model.forward(b)
        grad = self.loss_function(init_params, output, target)

        return grad