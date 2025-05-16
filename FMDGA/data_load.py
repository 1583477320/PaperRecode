from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import torch
import struct
import random
import os
import numpy as np
import torch
from torchvision import datasets, transforms
from PIL import Image

'''
数据生成，将mnist数据拼接
'''
# 读取MNIST图像和标签的函数
def generate_multi_mnist(num_samples, output_dir, train=True,save_images=False):
    """生成 MultiMNIST 数据集"""
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载 MNIST 数据集
    mnist = datasets.MNIST(
        root='./Mnist',
        train=train,
        download=False,
        transform=transforms.Compose([
            transforms.Pad(4, fill=0, padding_mode='constant'),  # 将图片调整为 36x36
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )

    # 将数据集按类别分组
    class_images = {i: [] for i in range(10)}
    for img, label in mnist:
        class_images[label].append(img)

    # 初始化图片和标签列表
    images = []
    labels = []

    # 生成 num_samples 个样本
    for i in range(num_samples):
        # 随机选择两个不同的类别
        label1 = np.random.randint(0, 10)
        label2 = label1
        while label2 == label1:
            label2 = np.random.randint(0, 10)

        # 随机选择两个不同类别的图片
        img1 = class_images[label1][np.random.randint(0, len(class_images[label1]))]
        img2 = class_images[label2][np.random.randint(0, len(class_images[label2]))]

        # 将图片转换为 numpy 数组
        img1_np = img1.numpy().squeeze()
        img2_np = img2.numpy().squeeze()

        # 生成 36x36 的画布
        canvas = np.zeros((36, 36), dtype=np.float32)

        # 随机移动 img1 和 img2，最多四个像素
        def random_shift(img):
            if img is img1_np:
                shift_x = np.random.randint(-6, 0)
                shift_y = np.random.randint(0, 6)
                shifted_img = np.roll(img, shift_x, axis=1)
                shifted_img = np.roll(shifted_img, shift_y, axis=0)
            else:
                shift_x = np.random.randint(0, 6)
                shift_y = np.random.randint(-6, 0)
                shifted_img = np.roll(img, shift_x, axis=1)
                shifted_img = np.roll(shifted_img, shift_y, axis=0)
            return shifted_img

        img1_shifted = random_shift(img1_np)
        img2_shifted = random_shift(img2_np)

        # 叠加两张图片
        combined_img = np.minimum(img1_shifted + img2_shifted, 1.0)  # 防止像素值超过 1.0

        # 将图片转换回 Tensor
        combined_img = torch.from_numpy(combined_img).unsqueeze(0)

        # 保存图像和标签
        if save_images:
            img_path = os.path.join(output_dir, f"sample_{i}.png")
            label_path = os.path.join(output_dir, f"sample_{i}_label.txt")
            Image.fromarray((combined_img.numpy().squeeze() * 255).astype(np.uint8)).save(img_path)
            with open(label_path, 'w') as f:
                f.write(f"{label1},{label2}")

        # 将图片和标签添加到对应列表中
        images.append(combined_img)
        labels.append((label1, label2))

    # 返回图片列表和标签列表
    return images, labels

# 数据初始化
class CompositeDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images  # 归一化到[0,1]
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # image = torch.tensor(self.images[idx].reshape(36 * 36))  # 展平为1296维
        image = torch.tensor(self.images[idx])  # 利用卷积
        label1 = torch.tensor(self.labels[idx][0], dtype=torch.long)
        label2 = torch.tensor(self.labels[idx][1], dtype=torch.long)
        return image, (label1, label2)


# ================== 数据分配函数 ==================
def split_data_to_servers(dataset, num_servers=5):
    """将数据集均匀分配到多个服务器"""
    total_size = len(dataset)
    indices = np.random.permutation(total_size)  # 随机打乱
    chunk_size = total_size // num_servers

    server_data = {}
    for i in range(num_servers):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i != num_servers - 1 else total_size
        server_data[f"server_{i}"] = Subset(dataset, indices[start:end])

    return server_data


# 加载原始数据集
class DuplicatedLabelMNIST(Dataset):
    def __init__(self, root, train=True):
        # 加载原始 MNIST 数据集
        self.original_dataset = datasets.MNIST(root='./Mnist', train=train, download=False, transform=transforms.Compose([
            transforms.Pad(4, fill=0, padding_mode='constant'),  # 将图片调整为 36x36
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )
        # 复制标签：将每个标签转换为 [label, label]
        self.labels = torch.stack([self.original_dataset.targets,
                                  self.original_dataset.targets], dim=1)
        self.images = self.original_dataset.data

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # 直接通过 original_dataset 获取图像（已应用 transform）
        image, _ = self.original_dataset[idx]  # image 已转换为 Tensor
        label = self.labels[idx]  # 标签为 [2] 的 Tensor
        return image, label

if __name__ == "__main__":
    # 创建可加载的 Dataset
    train_dataset = DuplicatedLabelMNIST(root='./data', train=True)
    test_dataset = DuplicatedLabelMNIST(root='./data', train=False)

    # 使用 DataLoader 加载数据
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 验证结果
    for index, (data, labels) in enumerate(train_loader):
        print("图像形状:", data.shape)  # 输出: torch.Size([64, 1, 28, 28])
        print("标签形状:", labels.shape)  # 输出: torch.Size([64, 2])
        print("示例标签:", labels[0])      # 例如: tensor([5, 5])
        break

# 使用示例
# 初始化生成器（使用训练集路径）
# generator = CompositeDatasetGenerator(
#     r"C:\Users\15834\PycharmProjects\paper_recode\OpenDataLab___MultiMNIST\raw\multi-mnist\data\train-images-idx3-ubyte",
#     r"C:\Users\15834\PycharmProjects\paper_recode\OpenDataLab___MultiMNIST\raw\multi-mnist\data\train-labels-idx1-ubyte"
# )
#
# # 生成一个批次
# batch_images, batch_labels = generator.generate_batch(16)
# print(batch_images[2])
# # 可视化验证
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(10, 10))
# for i in range(16):
#     plt.subplot(4, 4, i + 1)
#     plt.imshow(batch_images[i], cmap='gray')
#     plt.title(f"{batch_labels[i][0]}+{batch_labels[i][1]}")
#     plt.axis('off')
# plt.tight_layout()
# plt.show()
