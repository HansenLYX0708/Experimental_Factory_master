from tensorflow.keras import Sequential, layers
from CNN_plugin.Blur_pool import MaxBlurPooling2D
import tensorflow as tf

def create_classify_cnn(num_classes):
    model = Sequential()
    model.add(layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(num_classes))
    return model
