import cv2
import numpy as np
image=cv2.imread('3.png')
#cv2.imshow('original',image)
kernal_sharpening=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
sharpend=cv2.filter2D(image,-1,kernal_sharpening)
#cv2.imshow('image sharpening',sharpend)

cv2.imwrite("sharpness1.png", sharpend)
cv2.waitKey()
cv2.destroyAllWindows()