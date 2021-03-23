import mmap
import contextlib
import time
import numpy as np
import cv2

data = cv2.imread("C:/data/Slider_abs/A2/1-1-4x_0.bmp", 0)
test_file = mmap.mmap(-1, data.size, tagname='sharemem', access=mmap.ACCESS_WRITE)
test_file.write("1234567890".encode())

test_read = mmap.mmap(-1, data.size, tagname='sharemem', access=mmap.ACCESS_READ)
s = test_read.read().decode()

print("end")


np.memmap()