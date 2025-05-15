import torch.nn as nn
import torch

# 服务端模型（仅共享特征提取层）
class ServerSharedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(36 * 36, 256),
            nn.LeakyReLU()
        )

    # 可选：手动初始化参数
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.feature_extractor.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.feature_extractor(x)


# 定义初始化函数
