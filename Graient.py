from AggregateClients import aggregate_clients
from collections import OrderedDict
import torch.optim as optim

'''
服务端共享层更新梯度
'''


def federated_aggregation(server_model, client_gard, global_learn_rate=0.01):
    optimizer = optim.SGD(server_model.parameters(), lr=global_learn_rate, momentum=0.9)

    # 更新的服务端的梯度
    service_dict = aggregate_clients(client_gard)

    # 更新服务端参数
    server_model.zero_grad()
    for name, param in server_model.named_parameters():
        if name == "feature_extractor.1.weight":
            param.grad = service_dict[0]
        else:
            param.grad = service_dict[1]
    optimizer.step()

    return server_model
