import tensorflow_addons as tfa
import tensorflow as tf
from sklearn.metrics import roc_curve
import cv2
import timeit

a = [0.1, 1, 0.9999,  1]
#a[a > 0.5] = 1

from opencv.preprocess_abs_slider import preprocess_image, preprocess_simple

image_path = 'C:/data/From THO Basler Capture-selected/ABS A2/1-3-4x.bmp'
image = cv2.imread(image_path, 0)


start = timeit.default_timer()
preprocess_image(image, 'test', '', '')
end = timeit.default_timer()
print(end - start)

start1 = timeit.default_timer()
preprocess_simple(image, 'test', '', '')
end1 = timeit.default_timer()
print(end1 - start1)


print('end')
