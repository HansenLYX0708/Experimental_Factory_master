import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras.utils import plot_model
from PIL import Image
from sklearn import metrics


from classification.models.classifiers.create_model import create_classify_pt_cnn, create_classify_cnn,create_classify_pt_cnn_2
from classification.models.backbone.Inception import Inception
from classification.models.backbone.ResNet import ResNet
from classification.datasets.picker_tip_dataset import load_data, load_validation_data
from classification.models.callback import keras_callback
from classification.utils.plot_confusion_mat import plot_confusion_matrix

model_name = "My_model"
model_save_path = "model.h5"
log_base_path = os.path.join("logs\\")
model_output = os.path.join("logs\\best_model.h5")
weights_path = "weights\\model_weights.h5"
frozen_folder = "frozen_models"
frozen_name = "frozen_model.pb"
training = False
validation = True
load_weight = False
load_model = True
draw_Network_Graph = False
batch_size = 16
num_classes = 2
input_shape = (None, 256, 320, 1)
input_shape_for_blur = (256, 320, 1)
adamw_weight_delay = 1e-4
epochs = 10


if training:
    (x_train, y_train) = load_data()
    x_train = x_train.astype(np.float32) / 255.
    x_train = np.expand_dims(x_train, axis=3)
    # y_train = tf.one_hot(y_train, num_classes, dtype=np.float32)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=1050).batch(
        batch_size).prefetch(AUTOTUNE)

model = create_classify_pt_cnn_2(input_shape_for_blur, num_classes)
model.summary()

'''
model = ResNet([2, 2], 2)
model.build(input_shape=input_shape)
model.summary()
'''

if load_weight:
    model.load_weights(model_output)

if load_model:
    model = tf.keras.models.load_model(model_save_path)

if draw_Network_Graph:
    plot_model(model, model_name + ".png", show_shapes=True)



if training:
    optimizer = tfa.optimizers.AdamW(weight_decay=adamw_weight_delay)
    #optimizer = tf.optimizers.Adam()
    #loss = keras.losses.CategoricalCrossentropy(from_logits=True)
    loss = keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    callback = keras_callback.callback_setting(log_base_path, model_output, weight_only=True, if_log=True)

    history = model.fit(
        db_train,
        validation_data=db_train,
        epochs=epochs,
        shuffle=True,
        verbose=1,
        callbacks=callback)

    # save best model
    model.load_weights(model_output)
    model.save(model_save_path)

if validation:
    # data
    (x_train, y_train) = load_validation_data()
    x_train = x_train.astype(np.float32) / 255.
    x_train = np.expand_dims(x_train, axis=3)

    ret = model.predict(x_train)
    ret[ret > 0.5] = 1
    ret[ret <= 0.5] = 0
    confusion_matrix = metrics.confusion_matrix(y_train, ret)
    plot_confusion_matrix(confusion_matrix, 2)
    print(ret)
print('end')



