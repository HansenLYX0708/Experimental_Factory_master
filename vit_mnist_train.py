from classification.models.backbone import Inception
from classification.models.classifiers.ViT import VisionTransformer
import tensorflow_addons as tfa

import numpy as np
import tensorflow as tf
from tensorflow import keras

batch_size = 4096
training = True
load_weight = True
epochs = 10
weights_path = "model_weights.h5"

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train.astype(np.float32)/255., x_test.astype(np.float32)/255.
x_train, x_test = np.expand_dims(x_train, axis=3), np.expand_dims(x_test, axis=3)

db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

model = VisionTransformer(
        image_size=28,
        patch_size=4,
        num_layers=4,
        num_classes=10,
        d_model=64,
        num_heads=4,
        mlp_dim=128,
        channels=1,
        dropout=0.1,
    )

#model = Inception.Inception(2, 10)
model.build(input_shape=(None, 28, 28, 1))
model.summary()

#optimizer = keras.optimizers.Adam(learning_rate=1e-3)
optimizer = tfa.optimizers.AdamW(
        learning_rate=3e-4, weight_decay=1e-4
        )
criteon = keras.losses.CategoricalCrossentropy(from_logits=True)
acc_meter_train = keras.metrics.Accuracy()
acc_meter_test = keras.metrics.Accuracy()

#callback = keras_callback.callback_setting(log_base_path, model_output, True)

acc_record = 0


model.summary()
for epoch in range(1, epochs + 1):
    acc_meter_train.reset_states()
    if training :
        for step, (x, y) in enumerate(db_train):
            y_one_hot = tf.one_hot(y, depth=10)
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