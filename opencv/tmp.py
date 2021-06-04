import preprocess_abs_slider as preabs
import cv2
import os


def SingleTest():
    dir = "C:\\Users\\1000250081\Desktop\\New folder (2)\\New folder"
    name = "1622190648413.bmp"

    testimg = cv2.imread(os.path.join(dir, name), 0)
    preabs.preprocess_single_image(testimg, 'test.bmp')


if __name__ == '__main__':
    image_folder = "C:\\data\\slider_abs_6\\12"
    image_folder_save = "C:\\data\\slider_abs_6\\12_ROI"
    imgs = os.listdir(image_folder)
    for img in imgs:
        img_name = img.split('.')[0]
        image_path = os.path.join(image_folder, img)
        image = cv2.imread(image_path, 0)
        save_path = os.path.join(image_folder_save, img)
        preabs.preprocess_single_image(image, save_path)





