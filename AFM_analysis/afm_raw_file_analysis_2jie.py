import cv2
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from PIL import Image

# 加载 TIFF 文件
#img_path = "C:\\Users\\1000250081\\_work\\data\\AFM_Raw_file\\Excel sheet_Mild\\01_240723016\\01_B1E36&C12DOE1-F11\\NPTR\\R001_C003_01_Forward.tiff"
#img_path = "C:\\Users\\1000250081\\_work\\data\\AFM_Raw_file\\Excel sheet_Severe\\03_240419009\\01_B011E&C00_F1\\R001_C004_01_Forward.tiff"
#img_path = "C:\\Users\\1000250081\\_work\\data\\AFM_Raw_file\\Normal\\240426003\\01_B378B&C00_F1\\R001_C003_01_Forward.tiff"
img_path = "C:\\Users\\1000250081\\_work\\data\AFM_Raw_file\\4Normal\\062_R001_C001_01_Forward.tiff"
image = Image.open(img_path)
data = np.array(image)


# 定义二阶多项式函数
def quadratic_plane(XY, a, b, c, d, e, f):
    x, y = XY
    return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f

# 创建网格坐标
x = np.arange(data.shape[1])
y = np.arange(data.shape[0])
X, Y = np.meshgrid(x, y)

# 将网格展平用于拟合
x_flat = X.ravel()
y_flat = Y.ravel()
z_flat = data.ravel()

# 拟合二阶多项式
params, _ = curve_fit(quadratic_plane, (x_flat, y_flat), z_flat)
a, b, c, d, e, f = params

# 生成拟合曲面
fitted_surface = quadratic_plane((X, Y), a, b, c, d, e, f)

# 去除背景
flattened_data = data - fitted_surface

# 假设 'flattened_data' 是去除倾斜后的图像数据
flattened_data_normalized = (flattened_data - flattened_data.min()) / (flattened_data.max() - flattened_data.min()) * 255
flattened_data_normalized = flattened_data_normalized.astype(np.uint8)  # 转为 8-bit 数据
im_flattened_data = Image.fromarray(flattened_data_normalized)
im_flattened_data.save("1.tiff")


# 可视化结果
plt.subplot(1, 3, 1)
plt.title("Original Data")
plt.imshow(data, cmap="hot")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title("Fitted Surface")
plt.imshow(fitted_surface, cmap="hot")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title("Flattened Data")
plt.imshow(flattened_data, cmap="hot")
plt.colorbar()

plt.show()


