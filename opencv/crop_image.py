import cv2


if __name__ == '__main__':
    path = 'pick slider from tray_picker2_4.bmp'
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # 1030 , 218  1580 , 218
    # 1030 , 900  1580 , 900
    img2 = img[330:730, 500:900]
    cv2.imwrite('resizepot.bmp', img2)