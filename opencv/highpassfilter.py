import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#读取图像
img = cv.imread('C:\\Users\\1000250081\\_work\\data\\ufemto_slider\\39263698 Mark rej ok\\1-2.bmp', 0)


#傅里叶变换
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
fimg = np.log(np.abs(fshift))

#设置高通滤波器
rows, cols = img.shape
crow,ccol = int(rows/2), int(cols/2)
#fshift[0:crow-50, 0:ccol - 50] = 0
#fshift[0:crow-50, ccol + 50:cols] = 0
#fshift[crow + 50:rows,:ccol - 50] = 0
#fshift[crow + 50:rows:,ccol + 50:cols] = 0

#rows, cols = img.shape
#crow,ccol = int(rows/2), int(cols/2)
#mask = np.zeros((rows, cols, 2), np.uint8)
#mask[crow-30:crow+30, ccol-30:ccol+30] = 1



imgback = img[313:627, 1320:1786]

fb = np.fft.fft2(imgback)
fshiftb = np.fft.fftshift(fb)
cutfimg = np.log(np.abs(fshiftb))


dim = (img.shape[0], img.shape[1])

mask = np.zeros(dim)
mask[int((img.shape[0] - imgback.shape[0])/2):imgback.shape[0] + int((img.shape[0] - imgback.shape[0])/2),
     int((img.shape[1] - imgback.shape[1])/2):imgback.shape[1] + int((img.shape[1] - imgback.shape[1])/2)
    ] = fshiftb

#resizedf = cv2.copyMakeBorder(fshiftb, 10,10,10,10, cv2.BORDER_CONSTANT, value=0) #cv2.resize(cutfimg, dim)

fshiftnew = fshift - mask

#傅里叶逆变换
ishift = np.fft.ifftshift(fshiftnew)
iimg = np.fft.ifft2(ishift)
iimg = np.abs(iimg)
cv2.imwrite("fftimg.bmp", iimg)

#显示原始图像和高通滤波处理图像
plt.subplot(231), plt.imshow(img, 'gray'), plt.title('Original Image')
plt.axis('off')
plt.subplot(232), plt.imshow(fimg, 'gray'), plt.title('Original Image')
plt.axis('off')
plt.subplot(233), plt.imshow(imgback, 'gray'), plt.title('cut Image')
plt.axis('off')
plt.subplot(234), plt.imshow(cutfimg, 'gray'), plt.title('Result Image')
plt.axis('off')
plt.subplot(235), plt.imshow(iimg, 'gray'), plt.title('Result Image')
plt.axis('off')
plt.show()