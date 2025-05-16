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

    with torch.no_grad():
        client_model.feature_extractor.load_state_dict(server_weights)

    # 初始化梯度累积器：按任务存储共享层梯度
    grad_accumulator = {
        task_id: [torch.zeros_like(p) for p in client_model.feature_extractor.parameters()]
        for task_id in tasks
    }
    #记录损失
    task1_loss = []
    task2_loss = []
    for _ in tqdm(range(num_epochs)):
        task1_batch_loss = []
        task2_batch_loss = []
        for data, target in train_loader:
            data, target_task1, target_task2 = data.to(device), target.T[0].to(device), target.T[1].to(
                device)
            # total_grad = [torch.zeros_like(param) for param in client_model.feature_extractor.parameters()]
            # total_loss = 0

            # 分别计算每个任务的梯度并累积
            output1 = client_model(data)[0]
            output2 = client_model(data)[1]

            # 计算损失
            loss1 = criterion(output1, target_task1)
            loss2 = criterion(output2, target_task2)
            total_loss = loss1 + loss2

            '''
            手动累积梯度（如果需要记录各任务梯度）,分别查看每个任务对共享层的梯度,retain_graph=True：在反向传播时，设置 retain_graph=True 以保留计算图，否则计算图会在第一次反向传播后被释放，无法进行第二次反向传播。
            梯度的累加：如果你不冻结共享层的梯度，梯度会自动累加。因此，在分别计算每个任务的梯度时，需要先清除共享层的梯度。
            优化器的 zero_grad() 方法：在每次反向传播之前，建议使用 optimizer.zero_grad() 或手动清除梯度，以避免梯度累积。
            '''

            # 任务1的梯度
            for param in client_model.feature_extractor.parameters():
                param.grad = None  # 清除共享层的梯度
            loss1.backward(retain_graph=True)
            grad_accumulator["task1"][0] += client_model.feature_extractor[1].weight.grad.clone()
            grad_accumulator["task1"][1] += client_model.feature_extractor[1].bias.grad.clone()

            # 任务2的梯度
            for param in client_model.feature_extractor.parameters():
                param.grad = None  # 清除共享层的梯度
            loss2.backward(retain_graph=True)
            grad_accumulator["task2"][0] += client_model.feature_extractor[1].weight.grad.clone()
            grad_accumulator["task2"][1] += client_model.feature_extractor[1].bias.grad.clone()

            optimizer.zero_grad()
            # 单次反向传播（自动累加多任务梯度）
            total_loss.backward()  # loss1和loss2的梯度自动叠加

            # 更新参数
            optimizer.step()

            #记录损失
            task1_batch_loss.append(loss1.item())
            task2_batch_loss.append(loss2.item())
        task1_loss.append(sum(task1_batch_loss) / len(task1_batch_loss))
        task2_loss.append(sum(task2_batch_loss) / len(task2_batch_loss))
    # 计算每个任务的平均梯度
    avg_gradients = {}
    for task_id in tasks:
        avg_gradients[task_id] = [
            grad / num_epochs*len(train_loader) for grad in grad_accumulator[task_id]
        ]
    return client_model, avg_gradients, (sum(task1_loss) / len(task1_loss),sum(task2_loss) / len(task2_loss))

