import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
import tensorflow_addons as tfa


from classification.models.classifiers.create_model import create_classify_cnn
from classification.models.backbone.Inception import Inception
from classification.datasets.sn_data import load_data
from classification.models.callback import keras_callback


tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

model_save_path = "model.h5"
log_base_path = os.path.join("C:\\_work\\__project\\PyCharm_project\\Experimental_Factory_master\\logs\\")
model_output = os.path.join("C:\\_work\\__project\\PyCharm_project\\Experimental_Factory_master\\logs\\best_model.h5")
weights_path = "weights\\model_weights.h5"
frozen_folder = "frozen_models"
frozen_name = "frozen_model.pb"
log_base_path = os.path.join("logs\\")
model_output = os.path.join("logs\\best_model.h5")
training = False
load_weight = False
batch_size = 256
epochs = 100
input_shape = (None, 40, 24, 1)

# In[2]:

(x_train, y_train), (x_test, y_test) = load_data()
x_train, x_test = x_train.astype(np.float32)/255., x_test.astype(np.float32)/255.
x_train, x_test = np.expand_dims(x_train, axis=3), np.expand_dims(x_test, axis=3)
y_train = tf.one_hot(y_train, 16, dtype=np.float32)
y_test = tf.one_hot(y_test, 16, dtype=np.float32)
AUTOTUNE = tf.data.experimental.AUTOTUNE
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=84039).batch(batch_size).prefetch(AUTOTUNE)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).prefetch(AUTOTUNE)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# In[4]:



# build model and optimizer
model = create_classify_cnn(16)
model.build(input_shape=(None, 40, 24, 1))
model.summary()

if load_weight:
    model.load_weights(weights_path)

optimizer = tfa.optimizers.AdamW(weight_decay=1e-4)
loss = keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

callback = keras_callback.callback_setting(log_base_path, model_output, weight_only=False, if_log=True)

history = model.fit(
          db_train,
          validation_data=db_test,
          epochs=epochs,
          shuffle=True,
          verbose=1)
model.save(model_save_path)
print('end')
