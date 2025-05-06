from AggregateClients import aggregate_clients
from collections import OrderedDict

'''
服务端共享层更新梯度
'''


def federated_aggregation(server_model, client_gard, global_learn_rate):
    shared_parameters = list(server_model.feature_extractor.parameters())
    service_dict = aggregate_clients(client_gard)
    global_dict = server_model.state_dict()
    for key, gram in global_dict.items():
        if key == "feature_extractor.1.weight":
            global_dict[key] = service_dict[0]
        else:
            global_dict[key] = service_dict[1]

    # 将原来参数结构改写可和global_dict相加的结构
    client_grads_dict = OrderedDict()
    for (param_name, _), grad_tensor in zip(global_dict.items(), shared_parameters):
        client_grads_dict[param_name] = grad_tensor

    # 现在可以安全操作（如相加）
    update_shared_parameters = OrderedDict()
    for param_name in global_dict.keys():
        update_shared_parameters[param_name] = client_grads_dict[param_name] + global_learn_rate * global_dict[
            param_name]

    server_model.load_state_dict(update_shared_parameters)

    # 梯度置为0
    for param in server_model.feature_extractor.parameters():
        param.grad.zero_()

    return server_model
