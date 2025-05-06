from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import torch
import struct
import random

'''
数据生成，将mnist数据拼接
'''


# 读取MNIST图像和标签的函数
def read_mnist_images(path):
    with open(path, 'rb') as f:
        magic, size = struct.unpack('>II', f.read(8))
        rows, cols = struct.unpack('>II', f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(size, rows, cols)


def read_mnist_labels(path):
    with open(path, 'rb') as f:
        magic, size = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


# ==========核心叠加生成函数==========
def generate_composite_image(img1, img2):
    """
    生成复合图像的核心函数
    返回36x36的叠加图像和对应的位移信息
    """
    # 创建36x36画布
    canvas = np.zeros((36, 36), dtype=np.uint8)

    # 生成第一个数字的位移（-4到+4之间）
    dx1 = random.randint(-4, 4)
    dy1 = random.randint(-4, 4)

    # 生成第二个数字的位移（确保80%重叠）
    valid = False
    attempts = 0
    while not valid and attempts < 100:
        dx2 = random.randint(-4, 4)
        dy2 = random.randint(-4, 4)

        # 计算相对位移
        delta_x = dx2 - dx1
        delta_y = dy2 - dy1

        # 计算重叠面积
        w_overlap = 20 - abs(delta_x)
        h_overlap = 20 - abs(delta_y)
        if w_overlap > 0 and h_overlap > 0:
            overlap_area = w_overlap * h_overlap
            if overlap_area >= 320:  # 80% of 20x20
                valid = True
        attempts += 1

    # 如果无法找到有效位移，则使用相同位置
    if not valid:
        dx2 = dx1
        dy2 = dy1

    # 将数字放入画布（使用20x20中心区域）
    x1 = 8 + dx1  # 36/2 - 10 = 8
    y1 = 8 + dy1
    x2 = 8 + dx2
    y2 = 8 + dy2

    # 叠加第一个数字（取像素最大值）
    canvas[x1:x1 + 20, y1:y1 + 20] = np.maximum(canvas[x1:x1 + 20, y1:y1 + 20],
                                                img1[4:24, 4:24])  # 从28x28取中心20x20

    # 叠加第二个数字
    canvas[x2:x2 + 20, y2:y2 + 20] = np.maximum(canvas[x2:x2 + 20, y2:y2 + 20],
                                                img2[4:24, 4:24])

    return canvas


# 数据初始化
class CompositeDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images.astype(np.float32) / 255.0  # 归一化到[0,1]
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx].reshape(36 * 36))  # 展平为1296维
        label1 = torch.tensor(self.labels[idx][0], dtype=torch.long)
        label2 = torch.tensor(self.labels[idx][1], dtype=torch.long)
        return image, (label1, label2)


# ==========完整数据生成流程============
class CompositeDatasetGenerator:
    def __init__(self, image_path, label_path):
        self.images = read_mnist_images(image_path)
        self.labels = read_mnist_labels(label_path)

        # 预处理：为每个类别建立索引
        self.class_indices = {}
        for i in range(10):
            self.class_indices[i] = np.where(self.labels == i)[0]

    def generate_batch(self, batch_size):
        composite_images = []
        label_pairs = []

        for _ in range(batch_size):
            # 随机选择第一个数字
            idx1 = random.randint(0, len(self.images) - 1)
            label1 = self.labels[idx1]

            # 选择不同类别的第二个数字
            valid_labels = list(set(range(10)) - {label1})
            label2 = random.choice(valid_labels)
            idx2 = random.choice(self.class_indices[label2])

            # 生成复合图像
            composite = generate_composite_image(self.images[idx1],
                                                 self.images[idx2])

            composite_images.append(composite)
            label_pairs.append((label1, label2))

        return np.array(composite_images), np.array(label_pairs)


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

# # 使用示例
# # 初始化生成器（使用训练集路径）
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
