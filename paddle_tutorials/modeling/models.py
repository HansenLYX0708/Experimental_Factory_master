import paddle.nn as nn
from paddle.nn import Conv2D, ReLU, MaxPool2D, AdaptiveAvgPool2D, Linear,BatchNorm2D, Softmax

class MyModel(nn.Layer):
    def __init__(self):
        super(MyModel, self).__init__()
        self.in_channels = 1
        self.conv_1 = Conv2D(1, 16, 3)
        self.bn1 = BatchNorm2D(16)
        self.conv_2 = Conv2D(16, 32, 3)
        self.bn2 = BatchNorm2D(32)
        self.conv_3 = Conv2D(32, 64, 3)
        self.bn3 = BatchNorm2D(64)

        self.act1 = ReLU()
        self.act2 = ReLU()
        self.act3 = ReLU()

        self.pool1 = MaxPool2D(2)
        self.pool2 = MaxPool2D(2)
        self.avgpool = AdaptiveAvgPool2D(1)
        self.fc = Linear(64, 30)
        self.softmax = Softmax()
        self.flaten = nn.Flatten()

    def forward(self, inputs):
        x = self.conv_1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv_2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.conv_3(x)
        x = self.bn3(x)
        x = self.act3(x)

        x = self.avgpool(x)
        x = self.flaten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x