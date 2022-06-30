import cv2

img = cv2.imread("C:\\_work\\__document\\shan.jpg")
img = cv2.resize(img, (1600, 1130))
cv2.imwrite("resize.jpg", img)

img1 = cv2.imread("C:\\_work\\__document\\1.jpg")
img2 = cv2.imread("C:\\_work\\__document\\2.jpg")

img1 = cv2.resize(img1, (480, 680))

img2[1000:1000+680, 2217:2217+480] = img1

#img3 = cv2.add(img2, img2, mask=img1)

cv2.imwrite("merge.jpg",img2)