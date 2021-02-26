import os
import cv2
from matplotlib import pyplot as plt

def preprocess_single_image(image_name):
    # get image
    # convert to Gray
    image = cv2.imread(image_name, 0)

    # gaussian filter
    gaussian_image = cv2.GaussianBlur(src=image, ksize=(5, 5), sigmaX=1.5)
    #plt.subplot(131), plt.imshow(gaussian_image, "gray")

    # OTSU
    th1, otsu_image = cv2.threshold(gaussian_image, 0, 255, cv2.THRESH_OTSU)
    #plt.subplot(132), plt.imshow(otsu_image, "gray")

    # dilate
    dilate_img = cv2.dilate(otsu_image, (20, 20))
    #plt.subplot(133), plt.imshow(dilate_img, "gray")

    # Find contours
    contours = cv2.findContours(dilate_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # Min Rect


    # ROI

    # split

    #
    #plt.show()
    return

if __name__ == '__main__':
    image_path = "C:/Users/1000250081/Desktop/delete item/5-14-5x.bmp"
    preprocess_single_image(image_path)