from matplotlib import pyplot as plt
import numpy as np

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def relu(x):
    """relu函数"""
    # temp = np.zeros_like(x)
    # if_bigger_zero = (x > temp)
    # return x * if_bigger_zero
    return np.where(x < 0, 0, x)


def dx_relu(x):
    """relu函数的导数"""
    # temp = np.zeros_like(x)
    # if_bigger_equal_zero = (x >= temp)
    # return if_bigger_equal_zero * np.ones_like(x)
    return np.where(x < 0, 0, 1)

def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y
def silu(x):
    return x*sigmoid(x)
def dx_silu(x):
    return silu(x)+sigmoid(x)*(1-silu(x))


# ---------------------------------------------

if __name__ == '__main__':
    x = np.arange(-10, 10, 0.01)
    fx = relu(x)
    dx_fx = dx_relu(x)

    fx2 = silu(x)
    dx_fx2 = dx_silu(x)

    plt.subplot(1, 2, 1)
    ax = plt.gca()  # 得到图像的Axes对象
    ax.spines['right'].set_color('none')  # 将图像右边的轴设为透明
    ax.spines['top'].set_color('none')  # 将图像上面的轴设为透明
    ax.xaxis.set_ticks_position('bottom')  # 将x轴刻度设在下面的坐标轴上
    ax.yaxis.set_ticks_position('left')  # 将y轴刻度设在左边的坐标轴上
    ax.spines['bottom'].set_position(('data', 0))  # 将两个坐标轴的位置设在数据点原点
    ax.spines['left'].set_position(('data', 0))
    plt.title('Relu函数')
    plt.xlabel('x')
    plt.ylabel('fx')
    plt.plot(x, fx)
    plt.plot(x, fx2)

    plt.subplot(1, 2, 2)
    ax = plt.gca()  # 得到图像的Axes对象
    ax.spines['right'].set_color('none')  # 将图像右边的轴设为透明
    ax.spines['top'].set_color('none')  # 将图像上面的轴设为透明
    ax.xaxis.set_ticks_position('bottom')  # 将x轴刻度设在下面的坐标轴上
    ax.yaxis.set_ticks_position('left')  # 将y轴刻度设在左边的坐标轴上
    ax.spines['bottom'].set_position(('data', 0))  # 将两个坐标轴的位置设在数据点原点
    ax.spines['left'].set_position(('data', 0))
    plt.title('Relu函数的导数')
    plt.xlabel('x')
    plt.ylabel('dx_fx')
    plt.plot(x, dx_fx)
    plt.plot(x, dx_fx2)
    plt.show()
