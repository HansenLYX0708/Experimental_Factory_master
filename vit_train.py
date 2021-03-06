import os
from argparse import ArgumentParser

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import TensorBoard
from tensorflow import keras
import numpy as np

from classification.models.classifiers.ViT import VisionTransformer

AUTOTUNE = tf.data.experimental.AUTOTUNE

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--image-size", default=28, type=int)
    parser.add_argument("--patch-size", default=4, type=int)
    parser.add_argument("--num-layers", default=4, type=int)
    parser.add_argument("--d-model", default=64, type=int)
    parser.add_argument("--num-heads", default=4, type=int)
    parser.add_argument("--mlp-dim", default=128, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    args = parser.parse_args()

    ds = tfds.load("mnist", as_supervised=True)
    ds_train = (
        ds["train"]
        .cache()
        .shuffle(5 * args.batch_size)
        .batch(args.batch_size)
        #.prefetch(AUTOTUNE)
    )
    ds_test = (
        ds["test"]
        .cache()
        .batch(args.batch_size)
        #.prefetch(AUTOTUNE)
    )



    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.
    x_train, x_test = np.expand_dims(x_train, axis=3), np.expand_dims(x_test, axis=3)
    y_train, y_test = y_train.astype(np.float32), y_test.astype(np.float32)
    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(5 * args.batch_size).batch(256)
    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(256)

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    for step, (x, y) in enumerate(ds_train):
        break;

    for step, (x, y) in enumerate(db_train):
        break;

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = VisionTransformer(
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_layers=args.num_layers,
            num_classes=10,
            d_model=args.d_model,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            channels=1,
            dropout=0.1,
        )
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            optimizer=tfa.optimizers.AdamW(
                learning_rate=args.lr, weight_decay=args.weight_decay
            ),
            metrics=["accuracy"],
        )

    model.fit(
        ds_train,
        validation_data=ds_test,
        epochs=args.epochs,
        callbacks=[TensorBoard(log_dir=args.logdir, profile_batch=0),],
    )
    model.save_weights(os.path.join(args.logdir, "vit"))
