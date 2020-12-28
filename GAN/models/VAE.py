import tensorflow as tf
from tensorflow import keras

class VAE(tf.keras.Model):

    def __init__(self, h_dim, z_dim, image_size=28*28):
        super(VAE, self).__init__()

        # input => h
        self.fc1 = keras.layers.Dense(h_dim)
        # h => mu and variance
        self.fc2 = keras.layers.Dense(z_dim)
        self.fc3 = keras.layers.Dense(z_dim)

        # sampled z => h
        self.fc4 = keras.layers.Dense(h_dim)
        # h => image
        self.fc5 = keras.layers.Dense(image_size)

    def encode(self, x):
        h = tf.nn.relu(self.fc1(x))
        # mu, log_variance
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        """
        reparametrize trick
        :param mu:
        :param log_var:
        :return:
        """
        std = tf.exp(log_var * 0.5)
        eps = tf.random.normal(std.shape)

        return mu + eps * std

    def decode_logits(self, z):
        h = tf.nn.relu(self.fc4(z))
        return self.fc5(h)

    def decode(self, z):
        return tf.nn.sigmoid(self.decode_logits(z))

    def call(self, inputs, training=None, mask=None):
        # encoder
        mu, log_var = self.encode(inputs)
        # sample
        z = self.reparameterize(mu, log_var)
        # decode
        x_reconstructed_logits = self.decode_logits(z)

        return x_reconstructed_logits, mu, log_var