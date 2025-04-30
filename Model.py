import torch

class Model():
    def __init__(self, input_data, init_params):
        """

        :param input_data: 输入数据，矩阵
        :param init_params: 初始参数，向量
        """
        super(Model, self).__init__()

        self.input_data = input_data
        self.init_params = init_params

    def forward(self, b):
        """
        :param b: 偏置项
        :return: 函数的结果，向量
        """
        x =  self.input_data @ self.init_params.T + b

        return x