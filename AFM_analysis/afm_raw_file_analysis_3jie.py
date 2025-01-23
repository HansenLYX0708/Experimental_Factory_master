import numpy as np
import matplotlib.pyplot as plt
import tifffile


# 加载 TIFF 格式的 AFM 数据
def load_afm_tiff(file_path):
    return tifffile.imread(file_path)


# 三阶多项式拟合去除倾斜
def remove_tilt_and_flatten(data):
    # 获取数据的形状
    rows, cols = data.shape

    # 创建 x 和 y 网格
    x = np.linspace(0, cols - 1, cols)
    y = np.linspace(0, rows - 1, rows)
    X, Y = np.meshgrid(x, y)

    # 将数据重塑为一维数组，便于拟合
    x_data = X.flatten()
    y_data = Y.flatten()
    z_data = data.flatten()

    # 三阶拟合，获取倾斜面
    coefficients = np.polyfit(x_data, z_data, 3)  # 对 x 方向进行三阶拟合
    plane_fit_x = np.polyval(coefficients, x_data)

    # 将拟合得到的面重塑为与原数据同样的形状
    plane_fit_x = plane_fit_x.reshape(rows, cols)

    # 去除倾斜（平面化）
    data_flattened = data - plane_fit_x

    return data_flattened, plane_fit_x


# 示例：加载、去除倾斜并平面化
file_path = "C:\\Users\\1000250081\\_work\\data\\AFM_Raw_file\\1Mild\\001_R001_C003_01_Forward.tiff"
afm_data = load_afm_tiff(file_path)

# 去除倾斜并平面化
flattened_data, plane_fit = remove_tilt_and_flatten(afm_data)

# 可视化结果
plt.subplot(1, 2, 1)
plt.imshow(afm_data, cmap='hot')
plt.title('Original AFM Data')

plt.subplot(1, 2, 2)
plt.imshow(flattened_data, cmap='hot')
plt.title('Flattened AFM Data')

plt.colorbar()
plt.show()
