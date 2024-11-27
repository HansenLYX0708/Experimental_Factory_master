import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

img = Image.open(np.str('1658747610477.jpg')).convert('L')
img = np.array(img)
print(np.max(img))

print(np.min(img))
print((np.max(img)-np.min(img)) / (np.max(img)+np.min(img)))

print(img)



#plt.figure("black-white")
#plt.imshow(img, cmap='gray')
#plt.show()