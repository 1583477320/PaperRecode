import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
# -----------------客户端训练逻辑---------------
def client_local_train(client_model, server_weights, train_loader, tasks=["task1", "task2"], num_epochs=5, local_rate=0.01):
    # 客户端本地多任务训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client_model.train()
    optimizer = optim.SGD(client_model.parameters(), lr=local_rate)
    criterion = nn.CrossEntropyLoss()

    client_model.feature_extractor.load_state_dict(server_weights)

    # 初始化梯度累积器：按任务存储共享层梯度
    grad_accumulator = {
        task_id: {name: [] for name, param in client_model.named_parameters() if "feature_extractor" in name}
        for task_id in tasks
    }
    #记录损失
    task1_loss = []
    task2_loss = []
    for _ in tqdm(range(num_epochs)):
        task1_batch_loss = []
        task2_batch_loss = []
        for data, (target1, target2) in train_loader:
            data, target_task1, target_task2 = data.to(device), target1.to(device), target2.to(
                device)
            # total_grad = [torch.zeros_like(param) for param in client_model.feature_extractor.parameters()]
            # total_loss = 0

            # 分别计算每个任务的梯度并累积
            output1 = client_model(data)[0]
            output2 = client_model(data)[1]

            # 计算损失
            loss1 = criterion(output1, target_task1)
            loss2 = criterion(output2, target_task2)
            total_loss = (loss1 + loss2) / 2

            '''
            手动累积梯度（如果需要记录各任务梯度）,分别查看每个任务对共享层的梯度,retain_graph=True：在反向传播时，设置 retain_graph=True 以保留计算图，否则计算图会在第一次反向传播后被释放，无法进行第二次反向传播。
            梯度的累加：如果你不冻结共享层的梯度，梯度会自动累加。因此，在分别计算每个任务的梯度时，需要先清除共享层的梯度。
            优化器的 zero_grad() 方法：在每次反向传播之前，建议使用 optimizer.zero_grad() 或手动清除梯度，以避免梯度累积。
            '''

            # 任务1的梯度
            optimizer.zero_grad()  # 清除共享层的梯度
            loss1.backward(retain_graph=True)

            grads_task1 = {}
            for name, param in client_model.named_parameters():
                if param.grad is not None:
                    grads_task1[name] = param.grad.clone()

            # 任务2的梯度
            optimizer.zero_grad()  # 清除共享层的梯度
            loss2.backward(retain_graph=True)
            grads_task2 = {}
            for name, param in client_model.named_parameters():
                if param.grad is not None:
                    grads_task2[name] = param.grad.clone()

            optimizer.zero_grad()
            auto_up = True
            if auto_up:
                # # 单次反向传播（自动累加多任务梯度）
                total_loss.backward()  # loss1和loss2的梯度自动叠加
            else:
                # 手动合并梯度：共享层取均值，任务层保留各自的梯度
                for name, param in client_model.named_parameters():
                    if "feature_extractor" in name:  # 共享层
                        if name in grads_task1 and name in grads_task2:
                            # 共享层的梯度取均值
                            param.grad = (grads_task1[name] + grads_task2[name]) / 2
                    else:  # 任务层
                        # 任务层的梯度保留各自的梯度
                        if name in grads_task1:
                            param.grad = grads_task1[name]
                        elif name in grads_task2:
                            param.grad = grads_task2[name]

            # 梯度记录
            for name, param in client_model.named_parameters():
                if "feature_extractor" in name:  # 共享层
                    if name in grads_task1:
                        grad_accumulator['task1']["{}".format(name)].append(grads_task1[name])
                        grad_accumulator['task2']["{}".format(name)].append(grads_task2[name])
            # 更新参数
            optimizer.step()

            # 记录损失
            task1_batch_loss.append(loss1.item())
            task2_batch_loss.append(loss2.item())

        task1_loss.append(sum(task1_batch_loss) / len(task1_batch_loss))
        task2_loss.append(sum(task2_batch_loss) / len(task2_batch_loss))
    # 计算每个任务的平均梯度
    # 初始化平均结果字典
    avg_grad = average_tensor_lists(grad_accumulator)

    return client_model, avg_grad, (sum(task1_loss) / len(task1_loss),sum(task2_loss) / len(task2_loss))


def compute_average(tensor_list):
    if not tensor_list:
        return None  # 或者根据需求处理空列表的情况
    stacked = torch.stack(tensor_list, dim=0)
    avg = torch.mean(stacked, dim=0)
    return avg

def average_tensor_lists(input_dict):
    result = {}
    for task_name, shares in input_dict.items():
        new_shares = {}
        for share_name, tensor_list in shares.items():
            avg_tensor = compute_average(tensor_list)
            new_shares[share_name] = avg_tensor
        result[task_name] = new_shares
    return result