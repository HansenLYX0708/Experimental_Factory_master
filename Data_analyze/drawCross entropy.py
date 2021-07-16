import tensorflow
import matplotlib.pyplot as plt
import numpy as np

def cross_entropy_error(y,t):
    delta=1e-7  #添加一个微小值可以防止负无限大(np.log(0))的发生。
    return -(t*np.log(y+delta))

def cross_entropy_error_modulationfactor(y,t, m):
    delta=1e-7
    a = -(t*np.log(y+delta))
    return ((1-y)**m) * a



x = np.arange(0, 1, 0.01)

#y1 = cross_entropy_error(x, 1)
y2 = cross_entropy_error_modulationfactor(x, 1, 0)
y3 = cross_entropy_error_modulationfactor(x, 1, 0.5)
y4 = cross_entropy_error_modulationfactor(x, 1, 1)
y5 = cross_entropy_error_modulationfactor(x, 1, 1.5)
y6 = cross_entropy_error_modulationfactor(x, 1, 2)



#plt.title("cross entropy error with modulation factor")
#plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.plot(x, y4)
plt.plot(x, y5)
plt.plot(x, y6)
plt.legend(['y = 0', 'y = 0.5', 'y = 1', 'y = 1.5', 'y = 2'], loc='upper right')

plt.show()
