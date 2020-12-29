import tensorflow as tf
from tensorflow import keras

class ConvBNRelu(keras.Model):

    def __init__(self, ch, kernelsz=3, strides=1, padding='same'):
        super(ConvBNRelu, self).__init__()

        self.model = keras.models.Sequential([
            keras.layers.Conv2D(ch, kernelsz, strides=strides, padding=padding), #kernel_regularizer=keras.regularizers.l2(0.0001)),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU()
        ])

    def call(self, x, training=None):
        x = self.model(x, training=training)

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({

        })

class InceptionBlk(keras.Model):

    def __init__(self, ch, strides=1):
        super(InceptionBlk, self).__init__()

        self.ch = ch
        self.strides = strides

        self.conv1 = ConvBNRelu(ch, strides=strides)
        self.conv2 = ConvBNRelu(ch, kernelsz=3, strides=strides)
        self.conv3_1 = ConvBNRelu(ch, kernelsz=3, strides=strides)
        self.conv3_2 = ConvBNRelu(ch, kernelsz=3, strides=1)

        self.pool = keras.layers.MaxPooling2D(3, strides=1, padding='same')
        self.pool_conv = ConvBNRelu(ch, strides=strides)

    def call(self, x, training=None):
        x1 = self.conv1(x, training=training)

        x2 = self.conv2(x, training=training)

        x3_1 = self.conv3_1(x, training=training)
        x3_2 = self.conv3_2(x3_1, training=training)

        x4 = self.pool(x)
        x4 = self.pool_conv(x4, training=training)

        # concat along axis=channel
        x = tf.concat([x1, x2, x3_2, x4], axis=3)

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'channel' : self.ch,
            'strides' : self.strides

        })

class Inception(keras.Model):

    def __init__(self, num_layers, num_classes, init_ch=16, **kwargs):
        super(Inception, self).__init__(**kwargs)

        self.in_channels = init_ch
        self.out_channels = init_ch
        self.num_layers = num_layers
        self.init_ch = init_ch

        self.conv1 = ConvBNRelu(init_ch)

        self.blocks = keras.models.Sequential(name='dynamic-blocks')

        for block_id in range(num_layers):

            for layer_id in range(2):

                if layer_id == 0:

                    block = InceptionBlk(self.out_channels, strides=2)

                else:
                    block = InceptionBlk(self.out_channels, strides=1)

                self.blocks.add(block)

            # enlarger out_channels per block
            self.out_channels *= 2

        self.avg_pool = keras.layers.GlobalAveragePooling2D()
        self.fc = keras.layers.Dense(num_classes)

    def call(self, x, training=None):
        #out = keras.Sequential()
        #out.add(self.conv1)
        #out.add(self.blocks)
        #out.add(self.avg_pool)
        #out.add(self.fc)
        out = self.conv1(x, training=training)

        out = self.blocks(out, training=training)

        out = self.avg_pool(out)
        out = self.fc(out)

        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'in_channels':self.in_channels,
            'out_channels':self.out_channels,
            'init_channels':self.init_ch,
            'num_layers':self.num_layers,

        })
    # In[7]: