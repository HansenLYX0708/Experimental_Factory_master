import tensorflow_addons as tfa
import tensorflow as tf
from sklearn.metrics import roc_curve

a = [0.1, 1, 0.9999,  1]
a[a > 0.5] = 1


print('end')
