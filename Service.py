import torch.nn as nn


# 服务端模型（仅共享特征提取层）
class ServerSharedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(36 * 36, 256),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.feature_extractor(x)
