import tensorflow as tf
import numpy as np

class Squeeze_excitation_layer(tf.keras.Model):
    def __init__(self, filter_sq):
        # filter_sq 是 Excitation 中第一个全连接过程中的个数
        super().__init__()
        self.filter_sq = filter_sq
        self.avepool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(filter_sq)
        self.relu = tf.keras.layers.Activation('relu')
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, inputs):
        squeeze = self.avepool(inputs)

        excitation = self.dense(squeeze)
        excitation = self.relu(excitation)
        excitation = tf.keras.layers.Dense(inputs.shape[-1])(excitation)
        excitation = self.sigmoid(excitation)
        excitation = tf.keras.layers.Reshape((1, 1, inputs.shape[-1]))(excitation)

        scale = inputs * excitation

        return scale

if __name__ == '__main__':
    SE = Squeeze_excitation_layer(16)
    inputs = np.zeros((1, 32, 32, 32), dtype=np.float32)
    print(SE(inputs).shape)