import pandas as pd
import cv2
import numpy as np
import os
from classification.utils.csv_rwa import write_csv
from classification.datasets.sn_data import generateds
from classification.datasets.cv import augmentation_imgs_spesial

class PickerTipDataset(object):
    def __init__(self, csv_path, imagesfolder):
        self.name = ''
        self.csv_path = csv_path
        self.imagesfolder = imagesfolder
        self.x_train_image_name = []
        self.x_train = []
        self.y_train = []
        self.x_val = []
        self.y_val = []
        self.generate_data()

    def generate_data(self):
        # read data
        (x_train, y_train), (x_test, y_test) = self.load_data()
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
            src = pd.read_csv(self.csv_path, header=None)
        print(src.shape)
        src_df_shuffle = src.sample(frac=1)
        print(src_df_shuffle.shape)
        train_df = src_df_shuffle
        test_df = src_df_shuffle

        x_train = train_df.drop(train_df.columns[[1]], axis=1)
        y_train = train_df.drop(train_df.columns[[0]], axis=1)

        x_test = []
        y_test = []

        return (x_train.values, y_train.values), (x_test, y_test)

    def load_images(self):
        train_data = []
        for image in self.x_train_image_name:
            image_path = self.imagesfolder + image[0]
            image_data = cv2.imread(image_path, 0)
            image_data = cv2.resize(image_data, (320, 256))
            cv2.imwrite("a.bmp", image_data)
            image_data = image_data.tolist()

            image_data_exp = np.reshape(image_data, [320, 256, 1])
            train_data.append(image_data_exp)
        self.x_train = np.array(train_data)

def rename_preprocess(imgs_path, convert_path, index = 0):
    assert os.path.exists(imgs_path)
    files = os.listdir(imgs_path)

    for image in files:
        index = index + 1
        file_name = str(index) + ".bmp"
        image_path = os.path.join(imgs_path, image)
        image_data = cv2.imread(image_path, 0)
        height, width = image_data.shape
        resize_image = cv2.resize(image_data, (int(width / 4), int(height / 4)))

        cv2.imwrite(os.path.join(convert_path, file_name), resize_image)
    return

def create_class_csv(imgs_path, csv_path):
    assert os.path.exists(imgs_path)
    folders = os.listdir(imgs_path)
    data = []
    for folder in folders:
        single_class_folder = os.path.join(imgs_path, folder)
        assert os.path.exists(single_class_folder)
        imgs = os.listdir(single_class_folder)
        for img in imgs:
            img = os.path.join(folder, img)
            classes = 0
            if folder=='Good':
                classes = 1
            else:
                classes = 0
            data_row = [img, classes]
            #img_path = os.path.join(single_class_folder, img)

            #data_row = [img_path, folder]
            #data.append([img_data, folder])
            write_csv(csv_path, data_row)
    return

train_path = 'C:/data/picker_tip_convert/'  # 训练集输入特征路径
train_txt = 'C:/data/datasets_labels/picker_tip.csv'  # 训练集标签txt文件
x_train_savepath = 'C:/data/npy_file/pt_x_train.npy'  # 训练集输入特征存储文件
y_train_savepath = 'C:/data/npy_file/pt_y_train.npy'  # 训练集标签存储文件

def load_data():
    x_train_save = np.load(x_train_savepath)
    y_train = np.load(y_train_savepath)
    #x_train = np.reshape(x_train_save, (len(x_train_save), 28, 28))  # 将输入特征转换为28*28的形式
    #x_test = np.reshape(x_test_save, (len(x_test_save), 28, 28))  # 同上
    return (x_train_save, y_train)

if __name__ == '__main__':
    '''
    # convert tif to bmp,
    rename_preprocess('C:/data/picker_tip/Good', 'C:/data/picker_tip_convert/Good/')
    rename_preprocess('C:/data/picker_tip/NG', 'C:/data/picker_tip_convert/NG/')
    '''
    rename_preprocess('C:/data/picker_tip/validation', 'C:/data/picker_tip_augmentation/validation/')

    #picker_tip_ds = PickerTipDataset('C:/data/datasets_labels/picker_tip.csv',
    #                                 "C:/data/picker_tip_convert/")

    # Data Augmentation
    #augmentation_imgs_spesial("C:/data/picker_tip_convert/Good", "C:/data/picker_tip_augmentation/Good", 9)
    #augmentation_imgs_spesial("C:/data/picker_tip_convert/NG", "C:/data/picker_tip_augmentation/NG", 19)

    # Get class tag
    #create_class_csv('C:/data/picker_tip_convert/', 'C:/data/datasets_labels/picker_tip.csv')
    '''
    print('-------------Generate Datasets-----------------')
    x_train, y_train = generateds(train_path, train_txt)
    print('-------------Save Datasets-----------------')
    np.save(x_train_savepath, x_train)
    np.save(y_train_savepath, y_train)
    '''