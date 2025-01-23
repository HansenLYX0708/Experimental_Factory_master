import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import cm

# 加载 colormap 图像（如 'hot'）
image_path = "C:\\Users\\1000250081\\_work\\data\\AFM_Raw_file\\Normal\\240426003\\01_B378B&C00_F1\\R001_C003_01_Forward.tiff"

image = Image.open(image_path).convert("RGB")  # 转换为 RGB 模式
image_data = np.array(image)

# 获取 colormap 实例
colormap = cm.get_cmap('hot')

# 提取图像颜色值
# 将 RGB 值归一化到 [0, 1]
image_normalized = image_data / 255.0

# 反向查找 colormap 的数据值
# 由于 colormap 是连续的，可以用颜色距离最小化来近似映射
def rgb_to_data(rgb_array, cmap):
    """
    将归一化的 RGB 值映射回原始数据。
    rgb_array: 输入 RGB 数组，范围 [0, 1]
    cmap: colormap 实例 (matplotlib colormap)
    """
    # 获取 colormap 的归一化颜色值表
    cmap_colors = cmap(np.linspace(0, 1, 256))[:, :3]  # 取 RGB 值
    # 计算每个像素与 colormap 的距离
    distances = np.sqrt(((rgb_array[:, :, None, :] - cmap_colors[None, None, :, :]) ** 2).sum(axis=-1))
    # 找到距离最小的索引（对应数据值）
    data_indices = np.argmin(distances, axis=-1)
    # 映射回归一化数据值
    data_values = data_indices / 255.0
    return data_values

# 将图像 RGB 转化回数据
reconstructed_data = rgb_to_data(image_normalized, colormap)

# 可视化结果
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_data)

plt.subplot(1, 2, 2)
plt.title("Reconstructed Data")
plt.imshow(reconstructed_data, cmap='hot')
plt.colorbar()
plt.show()
