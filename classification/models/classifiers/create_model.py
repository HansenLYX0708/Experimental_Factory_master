from tensorflow.keras import Sequential, layers, regularizers, Input, Model
from CNN_plugin.Blur_pool import MaxBlurPooling2D, blur_pool
import tensorflow_addons as tfa
from CNN_plugin.Spatial_transformer import SpatialTransformer, spatial_yransform_network
from CNN_plugin.utils import get_initial_weights
from CNN_plugin.layers import BilinearInterpolation
from tensorflow import reshape

def create_classify_cnn(num_classes):
    model = Sequential()
    model.add(SpatialTransformer())
    model.add(layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu',
                            kernel_regularizer=regularizers.L2()))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu',
                            kernel_regularizer=regularizers.L2()))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                            kernel_regularizer=regularizers.L2()))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(filters=num_classes, kernel_size=1, strides=1, padding='same', activation='relu',
                            kernel_regularizer=regularizers.L2()))
    model.add(layers.GlobalAveragePooling2D())
    return model

def create_classify_cnn_2(input_shape, num_classes):
    input = Input(shape=input_shape)
    x = layers.BatchNormalization()(input)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization()(input)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(filters=num_classes, kernel_size=3, strides=1, padding="same", activation='relu')(x)
    output = layers.GlobalAvgPool2D()(x)
    model = Model(input, output)
    return model

def create_classify_cnn_3(input_shape, num_classes):
    input = Input(shape=input_shape)
    x = layers.BatchNormalization()(input)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation='relu')(x)


    res1 = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation='relu')(x)
    res1 = layers.MaxPool2D()(res1)
    res1 = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation='relu')(res1)
    x = layers.MaxPool2D()(x)
    x = x + res1

    res2 = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation='relu')(x)
    res2 = layers.MaxPool2D()(res2)
    res2 = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation='relu')(res2)
    x = layers.MaxPool2D()(x)
    x = x + res2

    x = layers.Conv2D(filters=num_classes, kernel_size=3, strides=1, padding="same", activation='relu')(x)
    output = layers.GlobalAvgPool2D()(x)
    model = Model(input, output)
    return model

def create_classify_pt_cnn(input_shape, num_classes):
    input = Input(shape=input_shape)
    x = layers.Conv2D(filters=16, kernel_size=3, strides=1, padding="same", activation='relu')(input)
    x = blur_pool(x)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation='relu')(x)
    x = blur_pool(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation='relu')(x)
    x = blur_pool(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation='relu')(x)
    x = blur_pool(x)
    #x = layers.Flatten()(x)
    #output = layers.Dense(num_classes)(x)
    x = layers.Conv2D(filters=num_classes, kernel_size=3, strides=1, padding="same", activation='relu')(x)
    output = layers.GlobalAvgPool2D()(x)
    model = Model(input, output)
    return model

def create_classify_pt_cnn_2(input_shape, num_classes):
    input = Input(shape=input_shape)
    '''
    locnet = layers.MaxPool2D(pool_size=(2, 2))(input)
    locnet = layers.Conv2D(20, (5, 5))(locnet)
    locnet = layers.MaxPool2D(pool_size=(2, 2))(locnet)
    locnet = layers.Conv2D(20, (5, 5))(locnet)
    locnet = layers.Flatten()(locnet)
    locnet = layers.Dense(50)(locnet)
    locnet = layers.Activation('relu')(locnet)
    weights = get_initial_weights(50)
    locnet = layers.Dense(6, weights=weights)(locnet)
    x = BilinearInterpolation((256, 320))([input, locnet])
    x = reshape(x, [-1, 256, 320, 1])
    '''
    x = layers.Conv2D(filters=8, kernel_size=3, padding="same", activation='relu')(input)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(filters=16, kernel_size=3, padding="same", activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(filters=32, kernel_size=3, padding="same", activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(filters=32, kernel_size=3, padding="same", activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1)(x)
    output = layers.Activation('sigmoid')(x)
    #x = layers.Conv2D(filters=num_classes, kernel_size=3, strides=1, padding="same", activation='sigmoid')(x)
    #output = layers.GlobalAvgPool2D()(x)

    model = Model(input, output)
    return model