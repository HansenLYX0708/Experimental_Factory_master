import pandas as pd
import cv2
import numpy as np
import os
from classification.utils.csv_rwa import write_csv

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
        cv2.imwrite(os.path.join(convert_path, file_name), image_data)
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

if __name__ == '__main__':
    # convert tif to bmp,
    #rename_preprocess('C:/data/picker_tip/Good', 'C:/data/picker_tip_convert/Good/')
    #rename_preprocess('C:/data/picker_tip/NG', 'C:/data/picker_tip_convert/NG/')

    # Get class tag
    #create_class_csv('C:/data/picker_tip_convert/', 'C:/data/datasets_labels/picker_tip.csv')

    picker_tip_ds = PickerTipDataset('C:/data/datasets_labels/picker_tip.csv',
                                     "C:/data/picker_tip_convert/")


