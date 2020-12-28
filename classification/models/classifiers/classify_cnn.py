import tensorflow as tf
from tensorflow.keras import layers, Sequential, losses, metrics

class Classify_CNN(tf.keras.Model):
    def __init__(self, num_classes):
        super(Classify_CNN, self).__init__()
        self.NUM_CLASSES = num_classes
        self.model = self.create_model()

    def call(self, inputs, training=True):
        if training:
            imgs, class_id = inputs
        else:
            imgs = inputs

        forward_result = self.model(imgs)

        if training:
            loss = losses.huber(class_id, forward_result)
            return loss
        else:
            return forward_result

    def create_model(self):
        model = Sequential()
        model.add(layers.Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation='relu'))
        model.add(layers.Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation='relu'))
        model.add(layers.Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation='relu'))
        model.add(layers.AveragePooling2D())
        model.add(layers.Dense(self.NUM_CLASSES))
        return model



