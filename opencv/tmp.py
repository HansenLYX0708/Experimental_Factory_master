import preprocess_abs_slider as preabs
import cv2

testimg = cv2.imread("1622173236269.bmp", 0)
preabs.preprocess_single_image(testimg, 'test.bmp')