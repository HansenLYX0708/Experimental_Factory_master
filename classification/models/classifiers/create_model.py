from tensorflow.keras import Sequential, layers, regularizers, Input, Model
from CNN_plugin.Blur_pool import MaxBlurPooling2D, blur_pool
import tensorflow_addons as tfa

def create_classify_cnn(num_classes):
    model = Sequential()
    model.add(layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu',
                            kernel_regularizer=regularizers.L2()))
    model.add(MaxBlurPooling2D())
    model.add(layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu',
                            kernel_regularizer=regularizers.L2()))
    model.add(tfa.layers.AdaptiveMaxPooling2D(output_size=(10, 6)))
    model.add(layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                            kernel_regularizer=regularizers.L2()))
    model.add(tfa.layers.AdaptiveMaxPooling2D(output_size=(5, 3)))
    model.add(layers.Conv2D(filters=num_classes, kernel_size=1, strides=1, padding='same', activation='relu',
                            kernel_regularizer=regularizers.L2()))
    model.add(tfa.layers.AdaptiveMaxPooling2D(output_size=(1, 1)))
    model.add(layers.GlobalAveragePooling2D())
    return model

def create_classify_cnn_2(input_shape, num_classes):
    input = Input(shape=input_shape)
    x = layers.Conv2D(filters=16, kernel_size=3, strides=1, padding="same", activation='relu')(input)
    x = blur_pool(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation='relu')(x)
    x = blur_pool(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation='relu')(x)
    x = blur_pool(x)
    x = layers.GlobalAvgPool2D()(x)
    output = layers.Dense(num_classes)(x)
    model = Model(input, output)
    return model