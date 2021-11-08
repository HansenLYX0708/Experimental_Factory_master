import tensorflow as tf
from classification.models.backbone.ResNet import ResNet
from classification.utils.tf_2_pb_to_frozen_graph import save_tf_2_frozen_graph, load_frozen_model_inference

import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train.astype(np.float32)/255., x_test.astype(np.float32)/255.
x_train, x_test = np.expand_dims(x_train, axis=3), np.expand_dims(x_test, axis=3)
y_train = tf.one_hot(y_train, 10, dtype=np.float32)
y_test = tf.one_hot(y_test, 10, dtype=np.float32)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(10000).batch(32)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# model_test = ResNet([2,2,2,2], 10)
# model_test.build(input_shape=(1, 28, 28, 1))
# tf.keras.utils.plot_model(model_test, 'absbs.png')
#
# input_shape = (None, 28, 28, 1)
# save_tf_2_frozen_graph(model_test,'', 'model.pb', input_shape)
#
#
# model_test.summary()


model = ResNet([2,2,2,2], 10)
optimizer = tf.optimizers.Adam()
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)



model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])


model.fit(train_dataset, validation_data=test_dataset, epochs=1, verbose=1)

