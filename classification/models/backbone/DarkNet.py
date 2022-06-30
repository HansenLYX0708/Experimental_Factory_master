import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow_addons as tfa


class ConvBNLayer(layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides = (1,1),
                 padding="valid",
                 groups = 1,
                 data_format='channels_last'):
        super(ConvBNLayer, self).__init__()

        self.conv = layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding=padding,
                                  groups=groups,
                                  data_format=data_format
                                  )
        self.batch_norm = layers.BatchNormalization()
        self.act = tfa.activations.mish

    def call(self, inputs):
        out = self.conv(inputs)
        out = self.batch_norm(out)
        out = self.act(out)
        return out


if __name__ == '__main__':
    input_shape = (4, 28, 28, 3)
    x = tf.ones(input_shape)
    conv_bn_layer = ConvBNLayer(1, 3)
    y = conv_bn_layer(x)

    print("done")

