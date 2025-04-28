import numpy as np
import cvxpy as cp
from typing import List, Dict, Any
from numpy import ndarray

'''
step1 任选60000张图片
step2 分M个客户端，每一个客户端 n= 60000/(2*M)个图片
step3 分两个任务，
step4 进行算法计算
step5 在算法第五步做两个任务的loss统计
'''

class Client:
    def __init__(self, objectives: List[callable], data: List[np.ndarray]):
        """
        客户端类，包含本地目标和数据
        :param objectives: 客户端的多目标函数列表
        :param data: 本地数据集（每个目标对应一个数据子集）
        """
        self.objectives = objectives
        self.data = data

    def compute_gradients(self, x, stochastic: bool = False, batch_size: int = 16) -> List[np.ndarray]:
        """
        计算本地梯度（全梯度或随机梯度）
        :param x: 当前每一个任务模型参数
        :param stochastic: 是否使用随机梯度
        :param batch_size: 随机梯度的批量大小
        :return: 各目标函数的梯度列表
        """
        gradients = []
        for a, obj in enumerate(self.objectives):
            if stochastic:
                # 随机采样批次数据
                idx = np.random.choice(len(self.data[a]), batch_size, replace=False)
                grad = obj(x[a], self.data[a][idx])
            else:
                # 全梯度计算
                grad = obj(x["local_params_{}".format(a)], self.data[a])
            gradients.append(grad)
        return gradients

class Server:
    def __init__(self, init_params: np.ndarray, num_clients: int):
        """
        服务器类，管理全局模型和聚合逻辑
        :param init_params: 初始模型参数
        :param num_clients: 客户端数量
        """
        self.global_params = init_params.copy()
        self.num_clients = num_clients

    def aggregate_count(self, client_updates) :
        """
        :param client_updates: 各客户端的目标梯度列表（client_updates[i][s]为第i个客户端的第s个目标梯度）
        :return: 所有任务的梯度列表
        """
        # 拼接所有目标的平均梯度
        # S = len(client_updates[0])  # 求对某任务感兴趣的客户端数量
        # 创建任务列表T
        all_keys = set()

        for inner_dict in client_updates.values():
            all_keys.update(inner_dict.keys())

        T = sorted(all_keys)

        # 空列表，用于储存服务器端每一个任务的梯度
        aggregated_grads = {}

        for s in T:  # 求对某任务感兴趣的客户端的平均梯度,并存入aggregated_grads
            grads_s = [inner_dict[s] for inner_dict in client_updates.values() if s in inner_dict]
            avg_grad = np.mean(grads_s, axis=0) if grads_s else np.zeros_like(self.global_params)
            aggregated_grads[s] = avg_grad

        return aggregated_grads

    @staticmethod
    def aggregate_updates(aggregated_grads: List[List[np.ndarray]]) -> np.ndarray:
        """
        聚合客户端的多目标梯度更新，求解二次规划确定全局下降方向
        :param aggregated_grads: 所有任务梯度列表
        :return: 全局下降方向向量d
        """
        # 拼接所有目标的平均梯度
        # S = len(client_updates[0])  # 求对某任务感兴趣的客户端数量

        # T = list(dict.fromkeys(client_updates[0]))    #任务列表
        # aggregated_grads = []     #空列表，用于储存服务器端每一个任务的梯度
        # for s in T:            #求对某任务感兴趣的客户端的平均梯度,并存入aggregated_grads
        #     grads_s = [client_updates[i][s] for i in client_updates[s]]
        #     avg_grad = np.mean(grads_s, axis=0) if grads_s else np.zeros_like(self.global_params)
        #     aggregated_grads.append(avg_grad)

        # 构建二次规划问题：求λ
        lambda_vars = cp.Variable(len(aggregated_grads))
        objective = cp.Minimize(
            cp.sum_squares(cp.sum([lambda_vars[s] * aggregated_grads[s] for s in range(len(aggregated_grads))], axis=0)))
        constraints = [cp.sum(lambda_vars) == 1, lambda_vars >= 0]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)

        # 计算全局下降方向
        d = np.sum([lambda_vars.value[s] * aggregated_grads[s] for s in range(len(aggregated_grads))], axis=0)
        return d

    def update_global_model(self, d: np.ndarray, eta: float) -> None:
        """按全局学习率更新模型参数"""
        self.global_params -= eta * d

class FMGDA:
    def __init__(self, clients: List[Client],
            server: Server,
            local_steps: int,
            eta_local: float,
            eta_global: float,
            stochastic: bool = False)-> np.ndarray:
        """
        FMGDA/FSMGDA主流程
        :param clients: 客户端列表
        :param server: 服务器
        :param num_rounds: 通信轮数
        :param local_steps: 每轮本地更新步数
        :param eta_local: 本地学习率
        :param eta_global: 全局学习率
        :param stochastic: 是否使用随机梯度（FSMGDA）
        :return: 最终全局模型参数
        """
        self.clients = clients
        self.server = server
        self.local_steps = local_steps
        self.eta_local = eta_local
        self.eta_global= eta_global
        self.stochastic = stochastic
    def backward(self) -> np.ndarray:
        # 1. 服务器广播全局模型
        current_params = server.global_params.copy()

        # 2. 客户端本地更新
        client_updates = {}
        for i, client in enumerate(clients):
            #客户端任务梯度更新
            client_test_updates = []

            #测试调整
            #初始化每一个任务的参数
            local_params_dic = {}

            for j in range(len(client.data)):
                local_params_dic["local_params_{}".format(i)] = current_params.copy()

            #返回该客户端所有的任务最终的梯度
            for _ in range(self.local_steps):
                grads = client.compute_gradients(local_params_dic) #返回的客户端所有任务该轮梯度列表
                # 返回每一个任务本轮次更新的参数
                for s in range(len(grads)):
                    local_params_dic["local_params_{}".format(s)] -= self.eta_local * grads[s]
                # 记录每一个任务每一轮梯度变化量（用于聚合）
                client_test_updates.append(grads)

            #转换为每一个任务对应的梯度列表
            merged_dict = {}
            for i, values in enumerate(zip(*client_test_updates)):
                merged_dict[i] = list(values)

            #返回每一个任务的最终梯度
            averages_dict = {key: sum(value)/len(value) for key, value in merged_dict.items()}
            #客户端本地更新
            client_updates[i] = averages_dict

        # 3. 服务器聚合并更新全局模型
        aggregate_list = Server.aggregate_count(client_updates=client_updates)

        d = server.aggregate_updates(list(aggregate_list.values()))
        server.update_global_model(d, self.eta_global)
        return server.global_params





#梯度计算函数

def gradient(x: np.ndarray, subset: np.ndarray = None) -> np.ndarray:
    # 示例梯度计算（假设为线性回归）
    if subset is None:
        subset = subset
    X, y = subset[:, :-1], subset[:, -1]
    return 2 * (X.T @ (X @ x - y)) / len(subset)



# 生成模拟数据

np.random.seed(42)

data_client1 = [np.random.randn(100, 11), np.random.randn(100, 11)]  # 客户端1的两个目标数据
data_client2 = [np.random.randn(100, 11), np.random.randn(100, 11)]  # 客户端2的两个目标数据
clients = [
    Client([gradient, gradient] ,data_client1),
    Client([gradient, gradient] ,data_client2)
]#客户端列表




# 初始化服务器
init_params = np.random.randn(10)
server = Server(init_params, num_clients=2)

# 运行FMGDA

FM_GDA = FMGDA(
    clients, server,
    local_steps=5,
    eta_local=0.01,
    eta_global=0.1,
    stochastic=False  # 设为True则使用FSMGDA
)

final_params = FM_GDA.backward()

print("Final global parameters:", final_params)