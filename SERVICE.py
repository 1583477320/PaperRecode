import numpy as np
import cvxpy as cp
from typing import List, Dict, Any
import Test

class SERVICE:
    def __init__(self, client_data:dict):
        """
        收集客户端梯度，然后进行服务端梯度计算
        :param client_data: 客户端梯度
        """
        self.client_data = client_data

    def service_compute_gradients(self):
        # 创建任务列表T
        all_keys = set()

        for inner_dict in self.client_data.values():
            all_keys.update(inner_dict.keys())

        T = sorted(all_keys)

        # 空列表，用于储存服务器端每一个任务的梯度
        aggregated_grads = {}
        for s in T:  # 求对某任务感兴趣的客户端的平均梯度,并存入aggregated_grads
            grads_s = [inner_dict[s] for inner_dict in self.client_data.values() if s in inner_dict]
            avg_grad = np.mean(grads_s, axis=0)
            aggregated_grads[s] = avg_grad

        # 构建二次规划问题：求λ
        lambda_vars = cp.Variable(len(aggregated_grads))
        objective = cp.Minimize(
            cp.sum_squares(
                cp.sum([lambda_vars[s] * aggregated_grads[s] for s in range(len(aggregated_grads))], axis=0)))
        constraints = [cp.sum(lambda_vars) == 1, lambda_vars >= 0]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)

        # 计算全局下降方向
        d = np.sum([lambda_vars.value[s] * aggregated_grads[s] for s in range(len(aggregated_grads))], axis=0)

        return d