import cv2
import numpy as np

img1 = cv2.imread("C:\\Users\\1000250081\\_work\\projects\\Experimental_Factory_master\\opencv\\drawpixel\\1-11__10.bmp", 0)
img2 = cv2.imread("C:\\Users\\1000250081\\_work\\projects\\Experimental_Factory_master\\opencv\\drawpixel\\1-11__30.bmp", 0)
img3 = cv2.imread("C:\\Users\\1000250081\\_work\\projects\\Experimental_Factory_master\\opencv\\drawpixel\\1-11__15.bmp", 0)
img4 = cv2.imread("C:\\Users\\1000250081\\_work\\projects\\Experimental_Factory_master\\opencv\\drawpixel\\1-11__20.bmp", 0)
img5 = cv2.imread("C:\\Users\\1000250081\\_work\\projects\\Experimental_Factory_master\\opencv\\drawpixel\\1-11__25.bmp", 0)

img6 = cv2.imread("C:\\Users\\1000250081\\_work\\projects\\Experimental_Factory_master\\opencv\\drawpixel2\\1-11_1.bmp", 0)
img7 = cv2.imread("C:\\Users\\1000250081\\_work\\projects\\Experimental_Factory_master\\opencv\\drawpixel2\\1-11_2.bmp", 0)
img8 = cv2.imread("C:\\Users\\1000250081\\_work\\projects\\Experimental_Factory_master\\opencv\\drawpixel2\\1-11_3.bmp", 0)
img9 = cv2.imread("C:\\Users\\1000250081\\_work\\projects\\Experimental_Factory_master\\opencv\\drawpixel2\\1-11_4.bmp", 0)

a1 = np.asarray(img1)
a2 = np.asarray(img2)
a3 = np.asarray(img3)
a4 = np.asarray(img4)
a5 = np.asarray(img5)
a6 = np.asarray(img6)
a7 = np.asarray(img7)
a8 = np.asarray(img8)
a9 = np.asarray(img9)

x = 51
y = 867

b1 = a1[867:868, 51:74]
b2 = a2[867:868, 51:74]
b3 = a3[867:868, 51:74]
b4 = a4[867:868, 51:74]
b5 = a5[867:868, 51:74]
b6 = a6[867:868, 51:74]
b7 = a7[867:868, 51:74]
b8 = a8[867:868, 51:74]
b9 = a9[867:868, 51:74]

print(b6)
print(b7)
print(b8)
print(b9)






