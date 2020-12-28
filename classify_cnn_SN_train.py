from classification.models.backbone.Inception import Inception
from classification.datasets.sn_data import load_data
from classification.models.callback import keras_callback
import numpy as np
import tensorflow as tf
import os
from tensorflow import keras

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')



# In[2]:


(x_train, y_train), (x_test, y_test) = load_data()
x_train, x_test = x_train.astype(np.float32)/255., x_test.astype(np.float32)/255.
#y_train, y_test = y_train.astype(np.float32), y_test.astype(np.float32)
# [b, 28, 28] => [b, 28, 28, 1]
x_train, x_test = np.expand_dims(x_train, axis=3), np.expand_dims(x_test, axis=3)

y_train_ohe = tf.one_hot(y_train, depth=16).numpy()
y_test_ohe = tf.one_hot(y_test, depth=16).numpy()

#db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=14784).batch(256)
#db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(256)

# In[3]:


print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# In[4]:



# build model and optimizer
batch_size = 32
epochs = 100
model = Inception(2, 16)
# derive input shape for every layers.
model.build(input_shape=(None, 40, 24, 1))#(None, 40, 24, 1)
model.summary()

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
#criteon = keras.losses.CategoricalCrossentropy(from_logits=True)
loss = keras.losses.CategoricalCrossentropy(from_logits=True)

#acc_meter = keras.metrics.Accuracy()

model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

log_base_path = os.path.join("C:\\_work\\__project\\PyCharm_project\\Experimental_Factory_master\\logs\\")
model_output = os.path.join("C:\\_work\\__project\\PyCharm_project\\Experimental_Factory_master\\logs\\best_model.h5")

callback = keras_callback.callback_setting(log_base_path, model_output, weight_only=False, if_log=True)

model.fit(x_train,
          y_train_ohe,
          validation_data=(x_test, y_test_ohe),
          validation_batch_size=256,

          batch_size=256,
          epochs=1,
          shuffle=True,
          verbose=1,
          callbacks=callback)

scores = model.call(x_train[:10], training=False)


#model.save_weights("model_weight.h5")
model.save("model.h5")