import cv2
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from PIL import Image

# 加载 TIFF 文件
img_path = "C:\\Users\\1000250081\\_work\\data\\AFM_Raw_file\\Excel sheet_Mild\\01_240723016\\01_B1E36&C12DOE1-F11\\NPTR\\R001_C003_01_Forward.tiff"
img_path = "C:\\Users\\1000250081\\_work\\data\\AFM_Raw_file\\Excel sheet_Severe\\03_240419009\\01_B011E&C00_F1\\R001_C004_01_Forward.tiff"
img_path = "C:\\Users\\1000250081\\_work\\data\\AFM_Raw_file\\Normal\\240426003\\01_B378B&C00_F1\\R001_C001_01_Forward.tiff"
image = Image.open(img_path)
data = np.array(image)

# 定义平面拟合函数
def plane(x, y, a, b, c):
    return a*x + b*y + c

# 创建坐标网格
x = np.arange(data.shape[1])
y = np.arange(data.shape[0])
X, Y = np.meshgrid(x, y)

# 拟合平面
params, _ = curve_fit(lambda XY, a, b, c: plane(XY[0], XY[1], a, b, c), (X.ravel(), Y.ravel()), data.ravel())
a, b, c = params

# 生成拟合平面
fitted_plane = plane(X, Y, a, b, c)

# 去除背景
flattened_data = data - fitted_plane

# 假设 'flattened_data' 是去除倾斜后的图像数据
flattened_data_normalized = (flattened_data - flattened_data.min()) / (flattened_data.max() - flattened_data.min()) * 255
flattened_data_normalized = flattened_data_normalized.astype(np.uint8)  # 转为 8-bit 数据
im_flattened_data = Image.fromarray(flattened_data_normalized)
im_flattened_data.save("1.tiff")

# 显示结果
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(data, cmap="viridis")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Flattened Image")
plt.imshow(flattened_data, cmap="viridis")
plt.colorbar()

plt.show()
