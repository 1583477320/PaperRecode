# 数据集介绍
  ## 简介
  MultiMNIST 数据集是从 MNIST 生成的。训练和测试是通过将一个数字叠加在来自同一组（训练或测试）但不同类别的另一个数字之上来生成的。每个数字在每个方向上最多移动 4 个像素，从而产生 36×36 的图像。考虑到 28×28 图像中的一个数字被限定在一个 20×20 的盒子中，两个数字边界框平均有 80% 的重叠。对于 MNIST 数据集中的每个数字，生成 1,000 个 MultiMNIST 示例，因此训练集大小为 60M，测试集大小为 10M。
  ## 引文
  ```
@article{sabour2017dynamic,
  title={Dynamic routing between capsules},
  author={Sabour, Sara and Frosst, Nicholas and Hinton, Geoffrey E},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```
  ‌​‌‌​​​​‌​​​‌‌‌‌‌​​‌‌​‌​‌​​‌​​​‌‌​‌‌‌​‌‌‌​​‌‌‌‌​‌​​​‌​‌‌‌​​‌‌‌‌​‌​‌‌​​‌‌‌​​‌‌‌‌​‌​​‌‌‌​‌