import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, UpSampling2D
import numpy as np
import tensorflow_addons as tfa


class ASPP(tf.keras.Model):

    def __init__(self, inputs_shape=(32, 32), depth = 256, **kwargs):
        super(ASPP, self).__init__(**kwargs)
        size = inputs_shape[2:]
        self.mean = tfa.layers.adaptive_pooling.AdaptiveAveragePooling2D((1, 1))
        self.conv = Conv2D(depth, 1, 1)
        self.upsample = UpSampling2D(size=size, interpolation="bilinear")
        self.atrous_block1 = Conv2D(depth, 1, 1)
        self.atrous_block6 = Conv2D(depth, 3, 1, padding="same", dilation_rate=6)
        self.atrous_block12 = Conv2D(depth, 3, 1, padding="same", dilation_rate=12)
        self.atrous_block18 = Conv2D(depth, 3, 1, padding="same", dilation_rate=18)
        self.conv_1X1_output = Conv2D(depth, 1, 1)


    def call(self, inputs):
        size = inputs.shape[2:]
        # 池化分支
        image_features = self.mean(inputs)
        image_features = self.conv(image_features)

        image_features = self.upsample(image_features)
        # 不同空洞率的卷积
        atrous_block1 = self.atrous_block1(inputs)
        atrous_block6 = self.atrous_block6(inputs)
        atrous_block12 = self.atrous_block12(inputs)
        atrous_block18 = self.atrous_block18(inputs)
        outputs = tf.concat([image_features, atrous_block1, atrous_block6, atrous_block12, atrous_block18], axis=1)
        # 利用1X1卷积融合特征输出
        outputs = self.conv_1X1_output(outputs)
        return outputs
    
    def get_config(self):
        config = super(ASPP, self).get_config()


if __name__ == '__main__':
    model = ASPP((None, 1, 32, 32))
    inputs = np.random.randint(0, 256, (3, 32, 32, 1))
    inputs = np.array(inputs, dtype=np.float32)
    outputs = model(inputs)
    outputs = outputs.numpy()
    print('end')
