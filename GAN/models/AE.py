import tensorflow as tf
from tensorflow import keras


class AE(tf.keras.Model):

    def __init__(self, h_dim = 20, image_size = 28*28):
        super(AE, self).__init__()

        # 784 => 512
        self.fc1 = keras.layers.Dense(512)
        # 512 => h
        self.fc2 = keras.layers.Dense(h_dim)

        # h => 512
        self.fc3 = keras.layers.Dense(512)
        # 512 => image
        self.fc4 = keras.layers.Dense(image_size)

    def encode(self, x):
        x = tf.nn.relu(self.fc1(x))
        x = (self.fc2(x))
        return x

    def decode_logits(self, h):
        x = tf.nn.relu(self.fc3(h))
        x = self.fc4(x)

        return x

    def decode(self, h):
        return tf.nn.sigmoid(self.decode_logits(h))

    def call(self, inputs, training=None, mask=None):
        # encoder
        h = self.encode(inputs)
        # decode
        x_reconstructed_logits = self.decode_logits(h)

        return x_reconstructed_logits