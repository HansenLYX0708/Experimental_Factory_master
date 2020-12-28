from classification.models.backbone.Inception import Inception
from classification.datasets.sn_data import load_data
from classification.models.callback import keras_callback
import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.python.framework import graph_util
from classification.utils.tf_2_pb_to_frozen_graph import save_tf_2_frozen_graph, load_frozen_model_inference

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


load_weights_path = "model_weight.h5"
log_base_path = os.path.join("C:\\_work\\__project\\PyCharm_project\\Experimental_Factory_master\\logs\\")
model_output = os.path.join("C:\\_work\\__project\\PyCharm_project\\Experimental_Factory_master\\logs\\best_model.h5")
weights_path = "weights\\model_weights.h5"
frozen_folder = "C:\\_work\\__project\\PyCharm_project\\Experimental_Factory_master\\frozen_models"
frozen_name = "frozen_model.pb"
training = True
load_weight = True
batch_size = 256
epochs = 10
input_shape = (None, 40, 24, 1)


(x_train, y_train), (x_test, y_test) = load_data()
x_train, x_test = x_train.astype(np.float32)/255., x_test.astype(np.float32)/255.
x_train, x_test = np.expand_dims(x_train, axis=3), np.expand_dims(x_test, axis=3)

db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=14784).batch(batch_size)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# build model and optimizer

model = Inception(2, 16)
model.build(input_shape=(None, 40, 24, 1))
model.summary()


if load_weight:
    model.load_weights(load_weights_path)

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
criteon = keras.losses.CategoricalCrossentropy(from_logits=True)
acc_meter_train = keras.metrics.Accuracy()
acc_meter_test = keras.metrics.Accuracy()

#callback = keras_callback.callback_setting(log_base_path, model_output, True)

acc_record = 0

for epoch in range(1, epochs + 1):
    acc_meter_train.reset_states()
    if training :
        for step, (x, y) in enumerate(db_train):
            y_one_hot = tf.one_hot(y, depth=16)
            with tf.GradientTape() as tape:
                # print(x.shape, y.shape)
                # [b, 10]
                logits = model(x)
                pred_train = tf.argmax(logits, axis=1)
                acc_meter_train.update_state(y, pred_train)
                # [b] vs [b, 10]
                loss = criteon(y_one_hot, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 10 == 0:
                print(epoch, step, 'loss:', loss.numpy(), 'acc:', acc_meter_train.result().numpy())
                acc_meter_train.reset_states()

    acc_meter_test.reset_states()
    for x, y in db_test:
        # [b, 10]
        logits = model(x, training=False)
        # [b, 10] => [b]
        pred = tf.argmax(logits, axis=1)
        # [b] vs [b, 10]
        acc_meter_test.update_state(y, pred)
    print(epoch, 'evaluation acc:', acc_meter_test.result().numpy())

    # save best model
    if acc_meter_test.result().numpy() > acc_record:
        print("save current model weight")
        model.save_weights(weights_path)

# at last, save most better frozen model
print("save best model frozen model")
model.load_weights(weights_path)
save_tf_2_frozen_graph(model, frozen_folder, frozen_name, input_shape)

load_frozen_model_inference(os.path.join(frozen_folder, frozen_name))

