import pandas as pd
import cv2
import numpy as np
import os
from classification.utils.csv_rwa import write_csv
import tensorflow as tf
from PIL import Image

class SliderSNDataset(object):
    def __init__(self, csv_path):
        self.name = ''
        self.csv_path = csv_path
        self.imagesfolder = ""
        self.x_train_image_name = []
        self.x_train = []
        self.y_train = []
        self.x_val = []
        self.y_val = []
        self.generate_data()

    def generate_data(self):
        # read data
        (x_train, y_train) = self.load_data()
        self.x_train_image_name = x_train
        self.y_train = y_train
        self.load_images()

    def normalization_data(self):
        return

    def split_data(self, rate):
        return

    def get_train_data(self):
        return self.x_train, self.y_train

    def get_val_data(self):
        return self.x_val, self.y_val

    def load_data(self):
        with open(self.csv_path, encoding='utf-8') as f:
            src = pd.read_csv(self.csv_path)
        print(src.shape)
        x_train = []
        y_train = []
        return (x_train.values, y_train.values)

    def load_images(self):
        self.imagesfolder = "C:/data/nearMems/nearMems_v1/"
        train_data = []
        for image in self.x_train_image_name:
            image_path = self.imagesfolder + image[0]
            image_data = cv2.imread(image_path, 0)
            #image_data = cv2.resize(image_data, (640, 512))
            image_data = image_data[145:785, 325:965]
            image_data = cv2.resize(image_data, (320, 320))
            #cv2.imwrite("a.bmp", image_data)
            image_data = image_data.tolist()

            image_data_exp = np.reshape(image_data, [320, 320, 1])
            train_data.append(image_data_exp)
        self.x_train = np.array(train_data)

def create_sn_csv(imgs_path, csv_path):
    assert os.path.exists(imgs_path)
    folders = os.listdir(imgs_path)
    data = []
    for folder in folders:
        single_class_folder = os.path.join(imgs_path, folder)
        assert os.path.exists(single_class_folder)
        imgs = os.listdir(single_class_folder)
        for img in imgs:
            img = os.path.join(folder, img)
            data_row = [img, folder]
            #img_path = os.path.join(single_class_folder, img)

            #data_row = [img_path, folder]
            #data.append([img_data, folder])
            write_csv(csv_path, data_row)

    return

train_path = 'C:/data/SliderSN/'    # 训练集输入特征路径
train_txt = 'C:/data/SliderSN.csv'  # 训练集标签txt文件
x_train_savepath = 'C:/data/SliderSN_mnist/sn_x_train.npy'   # 训练集输入特征存储文件
y_train_savepath = 'C:/data/SliderSN_mnist/sn_y_train.npy'   # 训练集标签存储文件

test_path = 'C:/data/SliderSN_test/'      # 测试集输入特征路径
test_txt = 'C:/data/SliderSN_test.csv'    # 测试集标签文件
x_test_savepath = 'C:/data/SliderSN_mnist/sn_x_test.npy'     # 测试集输入特征存储文件
y_test_savepath = 'C:/data/SliderSN_mnist/sn_y_test.npy'     # 测试集标签存储文件
'''
dict_label_to_id = {
    '0': 0,
    '2': 1,
    '3': 2,
    '4': 3,
    '5': 4,
    '6': 5,
    '7': 6,
    '8': 7,
    '9': 8,
    'A': 9,
    'B': 10,
    'C': 11,
    'D': 12,
    'E': 13,
    'F': 14,
    'L': 15,
}
'''
dict_label_to_id = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    'A': 10,
    'B': 11,
    'C': 12,
    'D': 13,
    'E': 14,
    'F': 15,
}

def load_data():
    x_train_save = np.load(x_train_savepath)
    y_train = np.load(y_train_savepath)
    x_test_save = np.load(x_test_savepath)
    y_test = np.load(y_test_savepath)
    #x_train = np.reshape(x_train_save, (len(x_train_save), 28, 28))  # 将输入特征转换为28*28的形式
    #x_test = np.reshape(x_test_save, (len(x_test_save), 28, 28))  # 同上
    return (x_train_save, y_train), (x_test_save, y_test)

def generateds(path, txt):      # 通过函数导入数据路径和
    f = open(txt, 'r')          # 以只读形式打开txt文件
    contents = f.readlines()    # 读取文件中所有行
    f.close()                   # 关闭txt文件
    x, y_ = [], []              # 建立空列表
    for content in contents:    # 逐行取出
        value = content.split(',')           # 以空格分开保存到value中，图片路径为value[0] , 标签为value[1] , 存入列表
        img_path = path + value[0]        # 拼接图片路径和文件名
        img = Image.open(img_path)        # 读入图片
        img = np.array(img.convert('L'))  # 图片变为8位宽灰度值的np.array格式
        #img = img / 255.                  # 数据归一化（实现预处理）
        x.append(img)                     # 归一化后的数据，贴到列表x
        y_value = dict_label_to_id[value[1].replace('\n', '')]
        y_.append(y_value)               # 标签贴到列表y_
        print('loading : ' + content)     # 打印状态提示

    x = np.array(x)           # x变为np.array格式
    y_ = np.array(y_)         # y变为np.array格式
    y_ = y_.astype(np.int64)  # y变为64位整型
    return x, y_              # 返回输入特征x，返回标签y_

if __name__ == '__main__':
    create_sn_csv('C:/data/SliderSN/', 'C:/data/SliderSN.csv')
    create_sn_csv('C:/data/SliderSN_test/', 'C:/data/SliderSN_test.csv')

    if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and os.path.exists(
            x_test_savepath) and os.path.exists(y_test_savepath):  # 判断训练集和测试集是否存在，是则直接读取
        print('-------------Load Datasets-----------------')
        x_train_save = np.load(x_train_savepath)
        y_train = np.load(y_train_savepath)
        x_test_save = np.load(x_test_savepath)
        y_test = np.load(y_test_savepath)
        x_train = np.reshape(x_train_save, (len(x_train_save), 28, 28))  # 将输入特征转换为28*28的形式
        x_test = np.reshape(x_test_save, (len(x_test_save), 28, 28))  # 同上
    else:  # 若数据集不存在，调用函数进行制作
        print('-------------Generate Datasets-----------------')
        x_train, y_train = generateds(train_path, train_txt)
        x_test, y_test = generateds(test_path, test_txt)
        print('-------------Save Datasets-----------------')
        # x_train_save = np.reshape(x_train, (len(x_train), -1))
        # x_test_save = np.reshape(x_test, (len(x_test), -1))
        np.save(x_train_savepath, x_train)
        np.save(y_train_savepath, y_train)
        np.save(x_test_savepath, x_test)
        np.save(y_test_savepath, y_test)

