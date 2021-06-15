import paddle.nn as nn
from paddle.nn import Conv2D, ReLU, MaxPool2D, AdaptiveMaxPool2D, Linear,BatchNorm2D

class MyModel(nn.Layer):
    def __init__(self):
        self.in_channels = 1
        self.conv_1 = Conv2D(1, 8, 3)
        self.bn = BatchNorm2D(8)
        self.act = ReLU()
        self.pool = MaxPool2D(2)
        



    def forward(self, *inputs, **kwargs):
        x = self.conv_1(inputs)



        return x