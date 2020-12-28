import tensorflow as tf
from tensorflow import keras

class my_loss(keras.losses.Loss):
    def __init__(self):
        super(my_loss, self).__init__()

    def call(self, y_true, y_pred):
        return

