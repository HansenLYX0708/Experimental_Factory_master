import os
import cv2
from matplotlib import pyplot as plt
import numpy as np

def preprocess_single_image(image, save_path):
    # get image
    # convert to Gray
    #image = cv2.imread(image_name, 0)

    # gaussian filter
    gaussian_image = cv2.GaussianBlur(src=image, ksize=(3, 3), sigmaX=1.5)
    #plt.subplot(131), plt.imshow(gaussian_image, "gray")

    # OTSU
    th1, otsu_image = cv2.threshold(gaussian_image, 0, 255, cv2.THRESH_OTSU)
    #plt.subplot(132), plt.imshow(otsu_image, "gray")

    # dilate
    kernel = np.ones((10, 10), np.uint8)
    dilate_img = cv2.dilate(otsu_image, kernel, iterations=1)
    #plt.subplot(133), plt.imshow(dilate_img, "gray")

    #cv2.imwrite("dilate.bmp", dilate_img)

    # Find contours
    contours, hierarchy = cv2.findContours(dilate_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #background = np.zeros((rows//2, cols//2), np.uint8)

    # Min Rect
    index = 0
    contour_len = cv2.arcLength(contours[0], True)
    for i in range(len(contours)):
        len_tmp = cv2.arcLength(contours[i], True)
        if len_tmp > contour_len:
            contour_len = len_tmp
            index = i
    #contours_img = cv2.drawContours(image, contours, index, (0, 255, 0), 2)
    #plt.imshow(contours_img, "gray")
    min_Rect = cv2.minAreaRect(contours[index])
    margn_Rect = []
    margn_Rect.append(min_Rect[0])
    margn_Rect.append([min_Rect[1][0] + 50, min_Rect[1][1] + 50])
    margn_Rect.append(min_Rect[2])

    # ROI
    width = int(margn_Rect[1][0])
    height = int(margn_Rect[1][1])
    angle = margn_Rect[2]
    if width < height:  # 计算角度，为后续做准备
        angle = angle - 90
    #print(angle)
    src_pts = cv2.boxPoints(tuple(margn_Rect))
    dst_pts = np.array([[0, height],
                        [0, 0],
                        [width, 0],
                        [width, height]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (width, height))

    '''
    if angle <= -90:  # 对-90度以上图片的竖直结果转正
        warped = cv2.transpose(warped)
        warped = cv2.flip(warped, 0)  # 逆时针转90度，如果想顺时针，则0改为1
    '''
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.transpose(warped)
        warped = cv2.flip(warped, 1)
    #plt.imshow(warped, "gray")
    #plt.show()
    #resize_warped = cv2.resize(warped, (1280, 1024), cv2.INTER_AREA)
    cv2.imwrite(save_path, warped)
    return

def preprocess_image(image, img_name, save_folder, label):
    rows = image.shape[0]
    cols = image.shape[1]

    images = []
    images.append(image[0:rows // 2, 0:cols // 2])
    images.append(image[0:rows // 2, cols // 2:cols])
    images.append(image[rows // 2:rows, 0:cols // 2])
    images.append(image[rows // 2:rows, cols // 2:cols])
    for i in range(len(images)):
        save_path = img_name + "_" + str(i) + ".bmp"
        save_path = os.path.join(save_folder, label, save_path)
        preprocess_single_image(images[i], save_path)

def preprocess_folder(image_folder, save_folder, label):
    assert os.path.exists(image_folder)
    images = os.listdir(image_folder)
    for img in images:
        img_name = img.split('.')[0]
        image_path = os.path.join(image_folder, img)
        image = cv2.imread(image_path, 0)
        preprocess_image(image, img_name, save_folder, label)


def preprocess_simple(image, img_name, save_folder, label):
    rows = image.shape[0]
    cols = image.shape[1]

    images = []
    images.append(image[0:rows // 2, 0:cols // 2])
    images.append(image[0:rows // 2, cols // 2:cols])
    images.append(image[rows // 2:rows, 0:cols // 2])
    images.append(image[rows // 2:rows, cols // 2:cols])
    return images

if __name__ == '__main__':
    image_path1 = "C:/data/From THO Basler Capture-selected/ABS A2/"
    image_path2 = "C:/data/From THO Basler Capture-selected/ABS A3&A5/"
    image_path3 = "C:/data/From THO Basler Capture-selected/ABS A8&A11/"
    image_path4 = "C:/data/From THO Basler Capture-selected/ABS M0/"

    save_folder = "C:/data/Slider_abs"
    preprocess_folder(image_path1, save_folder, "A2")
    preprocess_folder(image_path2, save_folder, "A3_A5")
    preprocess_folder(image_path3, save_folder, "A8_A11")
    preprocess_folder(image_path4, save_folder, "M0")