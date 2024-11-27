import cv2
import numpy as np


if __name__ == '__main__':
    # load image
    image_path = "C:\\Users\\1000250081\\Desktop\\WR Align\\Image__2023-09-19__15-40-33.bmp"

    src_img= cv2.imread(image_path)
    img_gray = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)

    cv2.namedWindow("original", 0)
    cv2.resizeWindow("original", 700, 900)
    cv2.imshow("original", src_img)

    img_gray_up = img_gray[1000:2000, 1000: 4000]
    cv2.namedWindow("img_gray_up", 0)
    cv2.resizeWindow("img_gray_up", 500, 500)
    cv2.imshow("img_gray_up", img_gray_up)
    cv2.imwrite("img_gray_up.bmp", img_gray_up)

    # sobel
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1)
    sobelx = cv2.convertScaleAbs(sobelx)  # 转回uint8
    sobely = cv2.convertScaleAbs(sobely)
    sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

    cv2.namedWindow("sobelxy", 0)
    cv2.resizeWindow("sobelxy", 700, 900)
    cv2.imshow("sobelxy", sobelxy)

    # OTSU
    # blur = cv2.GaussianBlur(sobelxy, (5, 5), 0)
    _, otsu_img = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

    cv2.namedWindow("otsu", 0)
    cv2.resizeWindow("otsu", 700, 900)
    cv2.imshow("otsu", otsu_img)
    cv2.imwrite("otsu.bmp", otsu_img)

    # open/close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    opening = cv2.morphologyEx(otsu_img, cv2.MORPH_OPEN, kernel, iterations=2)
    cv2.namedWindow("open", 0)
    cv2.resizeWindow("open", 700, 900)
    cv2.imshow('open', opening)
    cv2.imwrite("open.bmp", opening)

    erode = cv2.morphologyEx(opening, cv2.MORPH_ERODE, kernel, iterations=2)
    cv2.imwrite("erode.bmp", erode)
    #closing = cv2.morphologyEx(otsu_img, cv2.MORPH_CLOSE, kernel, iterations=2)
    #cv2.namedWindow("close", 0)
    #cv2.resizeWindow("close", 700, 900)
    #cv2.imshow('close', closing)

    # find contours
    contours, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 40000 and area > 20000:
            length = cv2.arcLength(contour, True)
            if length > 700 and length< 950:
                area_contours.append(contour)


    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     if area < 80000 and area > 40000:
    #         length = cv2.arcLength(contour, True)
    #         if length > 0:
    #             area_contours.append(contour)

    img_contour = cv2.drawContours(src_img, area_contours, -1, (0, 255, 0), 3)

    centers = []
    for area_contour in area_contours:
        rect = cv2.minAreaRect(area_contour)
        box = cv2.boxPoints(rect)
        centers.append((box[0] + box[2])/2)
        box = np.int0(box)
        cv2.drawContours(img_contour, [box], 0, (0, 0, 255), 2)

    linecenters = []
    linecenters.append((centers[0] + centers[1]) / 2)
    linecenters.append((centers[2] + centers[3]) / 2)

    centers = np.int0(centers)
    linecenters = np.int0(linecenters)
    cv2.line(img_contour, centers[0], centers[1], (0, 255, 255), 2)
    cv2.line(img_contour, centers[2], centers[3], (0, 255, 255), 2)
    cv2.line(img_contour, linecenters[0], linecenters[1], (255, 255, 0), 2)

    cv2.namedWindow("img_contour", 0)
    cv2.resizeWindow("img_contour", 700, 900)
    cv2.imshow('img_contour', img_contour)
    cv2.imwrite("img_withcounter.bmp", img_contour)



    cv2.waitKey()