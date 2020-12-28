import tensorflow as tf
import numpy as np


def load_image(image_file, is_train, IMG_HEIGHT, IMG_WIDTH):
    """
    load and preprocess images
    :param image_file:
    :param is_train:
    :return:
    """
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = image.shape[1]

    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    if is_train:
        # random jittering

        # resizing to 286 x 286 x 3
        input_image = tf.image.resize(input_image, [286, 286])
        real_image = tf.image.resize(real_image, [286, 286])

        # randomly cropping to 256 x 256 x 3
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
        input_image, real_image = cropped_image[0], cropped_image[1]

        if np.random.random() > 0.5:
            # random mirroring
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)
    else:
        input_image = tf.image.resize(input_image, size=[IMG_HEIGHT, IMG_WIDTH])
        real_image = tf.image.resize(real_image, size=[IMG_HEIGHT, IMG_WIDTH])

    # normalizing the images to [-1, 1]
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    # [256, 256, 3], [256, 256, 3]
    # print(input_image.shape, real_image.shape)

    # => [256, 256, 6]
    out = tf.concat([input_image, real_image], axis=2)

    return out

def generate_train_test_dataset(PATH):
    train_dataset = tf.data.Dataset.list_files(PATH + '/train/*.jpg')
    # The following snippet can not work, so load it hand by hand.
    # train_dataset = train_dataset.map(lambda x: load_image(x, True)).batch(1)
    train_iter = iter(train_dataset)
    train_data = []
    for x in train_iter:
        train_data.append(load_image(x, True))
    train_data = tf.stack(train_data, axis=0)
    # [800, 256, 256, 3]
    print('train:', train_data.shape)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
    train_dataset = train_dataset.shuffle(400).batch(1)

    test_dataset = tf.data.Dataset.list_files(PATH + 'test/*.jpg')
    # test_dataset = test_dataset.map(lambda x: load_image(x, False)).batch(1)
    test_iter = iter(test_dataset)
    test_data = []
    for x in test_iter:
        test_data.append(load_image(x, False))
    test_data = tf.stack(test_data, axis=0)
    # [800, 256, 256, 3]
    print('test:', test_data.shape)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
    test_dataset = test_dataset.shuffle(400).batch(1)
    return train_dataset, test_dataset


