import tensorflow as tf
from tensorflow import keras
import tensorflow_addons



l1_norm = tf.nn.l2_normalize()

l1_norm_keras = keras.regularizers.l2()
