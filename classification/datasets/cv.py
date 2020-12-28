import numpy as np
import cv2
from PIL import Image, ImageEnhance
import random
import os
import shutil

def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)

# 随机改变亮暗、对比度和颜色等
def random_distort(img):
    # 随机改变亮度
    def random_brightness(img, lower=0.8, upper=1.2):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Brightness(img).enhance(e)
    # 随机改变对比度
    def random_contrast(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Contrast(img).enhance(e)
    # 随机改变颜色
    def random_color(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Color(img).enhance(e)

    ops = [random_brightness, random_contrast, random_color]
    np.random.shuffle(ops)

    img = Image.fromarray(img)
    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)
    img = np.asarray(img)

    return img
def findAllFile_walk(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            yield f

def findAllFile_list(base):
    for f in os.listdir(base):
        yield f

def augmentation_imgs(imgs_folder, rate=1):
    # get path and
    imgs_list = os.listdir(imgs_folder)
    for i in range(rate):
        for f in imgs_list:
           input_img = cv2.imread(os.path.join(imgs_folder, f), cv2.IMWRITE_PAM_FORMAT_GRAYSCALE)
           output_img = random_distort(input_img)
           # get output name
           spl = f.split(".")
           output_img_name = spl[0] + "_" + str(i) + "." + spl[1]
           #output_folder = imgs_folder + "_aug"
           #create_dir_not_exist(output_folder)
           cv2.imwrite(os.path.join(imgs_folder, output_img_name), output_img)

def augmentation_imgs_spesial(imgs_folder, output_folder, rate=1):
    # clean up output folder
    shutil.rmtree(output_folder)
    os.mkdir(output_folder)
    # get path and


    imgs_list = os.listdir(imgs_folder)
    for i in range(rate):
        j = 0
        for f in imgs_list:
           input_img = cv2.imread(os.path.join(imgs_folder, f), cv2.IMWRITE_PAM_FORMAT_GRAYSCALE)
           output_img = random_distort(input_img)
           # get output name
           spl = f.split(".")
           output_img_name = spl[0] + "_" + str(i) + "." + spl[1]
           #output_folder = imgs_folder + "_aug"
           #create_dir_not_exist(output_folder)
           cv2.imwrite(os.path.join(output_folder, output_img_name), output_img)
           j=j+1
           if j > 100:
               break

def test1():
    input_path = "ocr_1_output_8.bmp"
    output_path = "ocr_1_output_8_1.bmp"
    input_img = cv2.imread(input_path, cv2.IMWRITE_PAM_FORMAT_GRAYSCALE)
    output_img = random_distort(input_img)

    cv2.imwrite(output_path, output_img)

def augmentation_all_cat():
    augmentation_imgs("D:/_work/_data/SliderSN/0", 3)
    augmentation_imgs("D:/_work/_data/SliderSN/2", 1)
    augmentation_imgs("D:/_work/_data/SliderSN/3", 4)
    augmentation_imgs("D:/_work/_data/SliderSN/4", 3)
    augmentation_imgs("D:/_work/_data/SliderSN/5", 7)
    augmentation_imgs("D:/_work/_data/SliderSN/6", 8)
    augmentation_imgs("D:/_work/_data/SliderSN/7", 23)
    augmentation_imgs("D:/_work/_data/SliderSN/8", 11)
    augmentation_imgs("D:/_work/_data/SliderSN/9", 2)
    augmentation_imgs("D:/_work/_data/SliderSN/A", 1)
    augmentation_imgs("D:/_work/_data/SliderSN/B", 27)
    augmentation_imgs("D:/_work/_data/SliderSN/C", 36)
    augmentation_imgs("D:/_work/_data/SliderSN/D", 13)
    augmentation_imgs("D:/_work/_data/SliderSN/E", 2)
    augmentation_imgs("D:/_work/_data/SliderSN/F", 15)
    augmentation_imgs("D:/_work/_data/SliderSN/L", 6)

def augmentation_all_cat_spesicial():
    augmentation_imgs_spesial("C:/data/SliderSN/0", "C:/data/SliderSN_test/0", 1)
    augmentation_imgs_spesial("C:/data/SliderSN/2", "C:/data/SliderSN_test/2", 1)
    augmentation_imgs_spesial("C:/data/SliderSN/3", "C:/data/SliderSN_test/3", 1)
    augmentation_imgs_spesial("C:/data/SliderSN/4", "C:/data/SliderSN_test/4", 1)
    augmentation_imgs_spesial("C:/data/SliderSN/5", "C:/data/SliderSN_test/5", 1)
    augmentation_imgs_spesial("C:/data/SliderSN/6", "C:/data/SliderSN_test/6", 1)
    augmentation_imgs_spesial("C:/data/SliderSN/7", "C:/data/SliderSN_test/7", 1)
    augmentation_imgs_spesial("C:/data/SliderSN/8", "C:/data/SliderSN_test/8", 1)
    augmentation_imgs_spesial("C:/data/SliderSN/9", "C:/data/SliderSN_test/9", 1)
    augmentation_imgs_spesial("C:/data/SliderSN/A", "C:/data/SliderSN_test/A", 1)
    augmentation_imgs_spesial("C:/data/SliderSN/B", "C:/data/SliderSN_test/B", 1)
    augmentation_imgs_spesial("C:/data/SliderSN/C", "C:/data/SliderSN_test/C", 1)
    augmentation_imgs_spesial("C:/data/SliderSN/D", "C:/data/SliderSN_test/D", 1)
    augmentation_imgs_spesial("C:/data/SliderSN/E", "C:/data/SliderSN_test/E", 1)
    augmentation_imgs_spesial("C:/data/SliderSN/F", "C:/data/SliderSN_test/F", 1)
    augmentation_imgs_spesial("C:/data/SliderSN/L", "C:/data/SliderSN_test/L", 1)


if __name__ == '__main__':
    #augmentation_all_cat()
    #augmentation_all_cat_spesicial()
    augmentation_imgs("C:/data/SliderSN/A", 14)


