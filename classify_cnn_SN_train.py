import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras.utils import plot_model

from classification.models.classifiers.create_model import create_classify_cnn, create_classify_cnn_2, create_classify_cnn_3
from classification.models.backbone.Inception import Inception
from classification.models.backbone.ResNet import ResNet
from classification.datasets.sn_data import load_data
from classification.models.callback import keras_callback
from classification.utils.tf_2_pb_to_frozen_graph import save_tf_2_frozen_graph

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

model_name = "My_model"
model_save_path = "model.h5"
log_base_path = os.path.join("logs\\")
model_output = os.path.join("logs\\best_model.h5")
weights_path = "weights\\model_weights.h5"
frozen_folder = "frozen_models"
frozen_name = "frozen_model.pb"
model_output = os.path.join("logs\\best_model.h5")
training = False
load_weight = False
draw_Network_Graph = True
save_frozen_model = False
batch_size = 256
num_classes = 16
input_shape = (None, 40, 24, 1)
adamw_weight_delay = 1e-4
epochs = 100


# In[2]:

(x_train, y_train), (x_test, y_test) = load_data()
x_train, x_test = x_train.astype(np.float32)/255., x_test.astype(np.float32)/255.
x_train, x_test = np.expand_dims(x_train, axis=3), np.expand_dims(x_test, axis=3)
y_train = tf.one_hot(y_train, num_classes, dtype=np.float32)
y_test = tf.one_hot(y_test, num_classes, dtype=np.float32)
AUTOTUNE = tf.data.experimental.AUTOTUNE
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=84039).batch(batch_size).prefetch(AUTOTUNE)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).prefetch(AUTOTUNE)

# In[4]:

# build model and optimizer
#model = ResNet([2, 2], num_classes)
#model.build(input_shape=input_shape)
#model = create_classify_cnn(num_classes)
model = create_classify_cnn(num_classes)
model.build(input_shape=input_shape)
model.summary()

if load_weight:
    model.load_weights(model_output)

if draw_Network_Graph:
    plot_model(model, model_name + ".png", show_shapes=True)

if training:
    optimizer = tfa.optimizers.AdamW(weight_decay=adamw_weight_delay)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    callback = keras_callback.callback_setting(log_base_path, model_output, weight_only=False, if_log=True)

    history = model.fit(
        db_train,
        validation_data=db_test,
        epochs=epochs,
        shuffle=True,
        verbose=1,
        callbacks=callback)

    model.save(model_save_path)

if save_frozen_model:
    # load best model
    model.load_weights(model_output)
    # save to frozen
    save_tf_2_frozen_graph(model, frozen_folder, frozen_name, input_shape)
print('end')
