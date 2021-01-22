# https://blog.csdn.net/weixin_41396062/article/details/104403655

import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import tensorflow as tf
import tensorflow.keras as keras


class MaxBlurPooling1D(Layer):

    def __init__(self, pool_size: int = 2, kernel_size: int = 3, **kwargs):
        self.pool_size = pool_size
        self.avg_kernel = None
        self.blur_kernel = None
        self.kernel_size = kernel_size

        super(MaxBlurPooling1D, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.kernel_size == 3:
            bk = np.array([2, 4, 2])
        elif self.kernel_size == 5:
            bk = np.array([6, 24, 36, 24, 6])
        else:
            raise ValueError

        bk = bk / np.sum(bk)
        bk = np.repeat(bk, input_shape[2])
        bk = np.reshape(bk, (self.kernel_size, 1, input_shape[2], 1))
        blur_init = keras.initializers.constant(bk)

        self.blur_kernel = self.add_weight(name='blur_kernel',
                                           shape=(self.kernel_size, 1, input_shape[2], 1),
                                           initializer=blur_init,
                                           trainable=False)

        super(MaxBlurPooling1D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):

        x = tf.nn.pool(x, (self.pool_size,), strides=(1,),
                       padding='SAME', pooling_type='MAX', data_format='NWC')

        x = K.expand_dims(x, axis=-2)
        x = K.depthwise_conv2d(x, self.blur_kernel, padding='same', strides=(self.pool_size, self.pool_size))
        x = K.squeeze(x, axis=-2)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], int(np.ceil(input_shape[1] / 2)), input_shape[2]


class MaxBlurPooling2D(Layer):

    def __init__(self, pool_size: int = 2, kernel_size: int = 3, **kwargs):
        self.pool_size = pool_size
        self.blur_kernel = None
        self.kernel_size = kernel_size

        super(MaxBlurPooling2D, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.kernel_size == 3:
            bk = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]])
            bk = bk / np.sum(bk)
        elif self.kernel_size == 5:
            bk = np.array([[1, 4, 6, 4, 1],
                           [4, 16, 24, 16, 4],
                           [6, 24, 36, 24, 6],
                           [4, 16, 24, 16, 4],
                           [1, 4, 6, 4, 1]])
            bk = bk / np.sum(bk)
        else:
            raise ValueError

        bk = np.repeat(bk, input_shape[3])

        bk = np.reshape(bk, (self.kernel_size, self.kernel_size, input_shape[3], 1))
        blur_init = keras.initializers.constant(bk)

        self.blur_kernel = self.add_weight(name='blur_kernel',
                                           shape=(self.kernel_size, self.kernel_size, input_shape[3], 1),
                                           initializer=blur_init,
                                           trainable=False)

        super(MaxBlurPooling2D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):

        x = tf.nn.pool(x, (self.pool_size, self.pool_size),
                       strides=(1, 1), padding='SAME', pooling_type='MAX', data_format='NHWC')
        x = K.depthwise_conv2d(x, self.blur_kernel, padding='same', strides=(self.pool_size, self.pool_size))

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], int(np.ceil(input_shape[1] / 2)), int(np.ceil(input_shape[2] / 2)), input_shape[3]

    def get_config(self):
        config = super(MaxBlurPooling2D, self).get_config()
        config.update({
            "pool_size":self.pool_size,
            "blur_kernel":self.blur_kernel,
            "kernel_size":self.kernel_size
        })


class AverageBlurPooling1D(Layer):

    def __init__(self, pool_size: int = 2, kernel_size: int = 3, **kwargs):
        self.pool_size = pool_size
        self.blur_kernel = None
        self.kernel_size = kernel_size

        super(AverageBlurPooling1D, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.kernel_size == 3:
            bk = np.array([2, 4, 2])
        elif self.kernel_size == 5:
            bk = np.array([6, 24, 36, 24, 6])
        else:
            raise ValueError

        bk = bk / np.sum(bk)
        bk = np.repeat(bk, input_shape[2])
        bk = np.reshape(bk, (self.kernel_size, 1, input_shape[2], 1))
        blur_init = keras.initializers.constant(bk)

        self.blur_kernel = self.add_weight(name='blur_kernel',
                                           shape=(self.kernel_size, 1, input_shape[2], 1),
                                           initializer=blur_init,
                                           trainable=False)

        super(AverageBlurPooling1D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):

        x = tf.nn.pool(x, (self.pool_size,), strides=(1,), padding='SAME', pooling_type='AVG',
                       data_format='NWC')
        x = K.expand_dims(x, axis=-2)
        x = K.depthwise_conv2d(x, self.blur_kernel, padding='same', strides=(self.pool_size, self.pool_size))
        x = K.squeeze(x, axis=-2)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], int(np.ceil(input_shape[1] / 2)), input_shape[2]


class AverageBlurPooling2D(Layer):

    def __init__(self, pool_size: int = 2, kernel_size: int = 3, **kwargs):
        self.pool_size = pool_size
        self.blur_kernel = None
        self.kernel_size = kernel_size

        super(AverageBlurPooling2D, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.kernel_size == 3:
            bk = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]])
            bk = bk / np.sum(bk)
        elif self.kernel_size == 5:
            bk = np.array([[1, 4, 6, 4, 1],
                           [4, 16, 24, 16, 4],
                           [6, 24, 36, 24, 6],
                           [4, 16, 24, 16, 4],
                           [1, 4, 6, 4, 1]])
            bk = bk / np.sum(bk)
        else:
            raise ValueError

        bk = np.repeat(bk, input_shape[3])

        bk = np.reshape(bk, (self.kernel_size, self.kernel_size, input_shape[3], 1))
        blur_init = keras.initializers.constant(bk)

        self.blur_kernel = self.add_weight(name='blur_kernel',
                                           shape=(self.kernel_size, self.kernel_size, input_shape[3], 1),
                                           initializer=blur_init,
                                           trainable=False)

        super(AverageBlurPooling2D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):

        x = tf.nn.pool(x, (self.pool_size, self.pool_size), strides=(1, 1), padding='SAME', pooling_type='AVG',
                       data_format='NHWC')
        x = K.depthwise_conv2d(x, self.blur_kernel, padding='same', strides=(self.pool_size, self.pool_size))

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], int(np.ceil(input_shape[1] / 2)), int(np.ceil(input_shape[2] / 2)), input_shape[3]


class BlurPool2D(Layer):
    def __init__(self, pool_size: int = 2, kernel_size: int = 3, **kwargs):
        self.pool_size = pool_size
        self.blur_kernel = None
        self.kernel_size = kernel_size

        super(BlurPool2D, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.kernel_size == 3:
            bk = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]])
            bk = bk / np.sum(bk)
        elif self.kernel_size == 5:
            bk = np.array([[1, 4, 6, 4, 1],
                           [4, 16, 24, 16, 4],
                           [6, 24, 36, 24, 6],
                           [4, 16, 24, 16, 4],
                           [1, 4, 6, 4, 1]])
            bk = bk / np.sum(bk)
        else:
            raise ValueError

        bk = np.repeat(bk, input_shape[3])

        bk = np.reshape(bk, (self.kernel_size, self.kernel_size, input_shape[3], 1))
        blur_init = keras.initializers.constant(bk)

        self.blur_kernel = self.add_weight(name='blur_kernel',
                                           shape=(self.kernel_size, self.kernel_size, input_shape[3], 1),
                                           initializer=blur_init,
                                           trainable=False)

        super(BlurPool2D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        x = K.depthwise_conv2d(x, self.blur_kernel, padding='same', strides=(self.pool_size, self.pool_size))

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], int(np.ceil(input_shape[1] / 2)), int(np.ceil(input_shape[2] / 2)), input_shape[3]


class BlurPool1D(Layer):

    def __init__(self, pool_size: int = 2, kernel_size: int = 3, **kwargs):
        self.pool_size = pool_size
        self.blur_kernel = None
        self.kernel_size = kernel_size

        super(BlurPool1D, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.kernel_size == 3:
            bk = np.array([2, 4, 2])
        elif self.kernel_size == 5:
            bk = np.array([6, 24, 36, 24, 6])
        else:
            raise ValueError

        bk = bk / np.sum(bk)
        bk = np.repeat(bk, input_shape[2])
        bk = np.reshape(bk, (self.kernel_size, 1, input_shape[2], 1))
        blur_init = keras.initializers.constant(bk)

        self.blur_kernel = self.add_weight(name='blur_kernel',
                                           shape=(self.kernel_size, 1, input_shape[2], 1),
                                           initializer=blur_init,
                                           trainable=False)

        super(BlurPool1D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):

        x = K.expand_dims(x, axis=-2)
        x = K.depthwise_conv2d(x, self.blur_kernel, padding='same', strides=(self.pool_size, self.pool_size))
        x = K.squeeze(x, axis=-2)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], int(np.ceil(input_shape[1] / 2)), input_shape[2]


def blur_pool(input):
    input_shape = input.shape
    kernel_size = 3
    pool_size = 2
    if kernel_size == 3:
        bk = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]])
        bk = bk / np.sum(bk)
    elif kernel_size == 5:
        bk = np.array([[1, 4, 6, 4, 1],
                       [4, 16, 24, 16, 4],
                       [6, 24, 36, 24, 6],
                       [4, 16, 24, 16, 4],
                       [1, 4, 6, 4, 1]])
        bk = bk / np.sum(bk)
    else:
        raise ValueError

    bk = np.repeat(bk, input_shape[3])

    bk = np.reshape(bk, (kernel_size, kernel_size, input_shape[3], 1))
    blur_init = keras.initializers.constant(bk)
    blur_kernel_layer = keras.layers.Layer()
    blur_kernel = blur_kernel_layer.add_weight(name='blur_kernel',
                                           shape=(kernel_size, kernel_size, input_shape[3], 1),
                                           initializer=blur_init,
                                           trainable=False)
    x = K.depthwise_conv2d(input, blur_kernel, padding='same', strides=(pool_size, pool_size))
    return x

if __name__ == '__main__':

    inputs = np.random.randint(0, 256, (3, 32, 32, 3))
    inputs = np.array(inputs, dtype=np.float32)
    model = MaxBlurPooling2D()
    outputs = model(inputs)
    outputs = outputs.numpy()

    output2 = blur_pool(inputs)

    print('end')