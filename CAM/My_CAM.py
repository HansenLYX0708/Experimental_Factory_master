# https://keras.io/examples/vision/grad_cam/
import numpy as np
import tensorflow as tf
from tensorflow import keras
from classification.models.backbone.ResNet import ResNet
from classification.models.classifiers.create_model import create_classify_cnn, create_classify_cnn_2
# Display
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from classification.utils.tf_2_pb_to_frozen_graph import load_frozen_model_inference

#[1]Configurable parameters
'''
You can change these to another model.
To get the values for last_conv_layer_name and classifier_layer_names, use model.summary() to see the names of all layers in the model.
'''

'''
model_builder = keras.applications.xception.Xception

preprocess_input = keras.applications.xception.preprocess_input
decode_predictions = keras.applications.xception.decode_predictions
'''
img_size = (40, 24)
last_conv_layer_name = "conv2d_1"
classifier_layer_names = [
    #"tf_op_layer_DepthwiseConv2dN",
    #"conv2d_1",
    #"tf_op_layer_DepthwiseConv2dN",
    "conv2d_2",
    #"tf_op_layer_DepthwiseConv2dN",
    "global_average_pooling2d",
    "dense",
]
'''
# The local path to our target image
img_path = keras.utils.get_file(
    "african_elephant.jpg", " https://i.imgur.com/Bvro0YD.png"
)
'''

img_path = "G:\\_work\\_data\\sliderSN_data\\SliderSN_test\\0\\ocr_108_output_8_2_0.bmp"
#display(Image(img_path))

#[2] Grad CAM
def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size, color_mode="grayscale")
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names
):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)

    #model_inputs = keras.Input(shape=[None, 40, 24, 1])
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


#[3] test drive
# Prepare image
img_array = get_img_array(img_path, size=img_size)

# Make model
#model = model_builder(weights="imagenet")
#model.summary()

load_weights_path = "G:\\github\\Experimental_Factory_master\\logs\\1_MaxPool\\best_model.h5"
#model = ResNet([2, 2, 2], 16)
#model = create_classify_cnn_2([40, 24, 1], 16)
model = create_classify_cnn_2(16)
model.build(input_shape=(None, 40, 24, 1))
model.summary()
model.load_weights(load_weights_path)

'''
frozen_folder = "G:\\github\\Experimental_Factory_master\\frozen_models"
frozen_name = "frozen_model.pb"
img_path = "G:\\_work\\_data\\sliderSN_data\\SliderSN_inference\\"
model = load_frozen_model_inference(os.path.join(frozen_folder, frozen_name))

model.summary()
'''
# Print what the top predicted class is
preds = model.predict(img_array)
# softmax
preds_softmax = tf.nn.softmax(preds)

print("Predicted:", preds_softmax)

# Generate class activation heatmap
heatmap = make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names
)

# Display heatmap
#plt.matshow(heatmap)
#plt.show()

#[4] Create a superimposed visualization
# We load the original image
img = keras.preprocessing.image.load_img(img_path)
img = keras.preprocessing.image.img_to_array(img)

# We rescale heatmap to a range 0-255
heatmap = np.uint8(255 * heatmap)

# We use jet colormap to colorize heatmap
jet = cm.get_cmap("jet")#

# We use RGB values of the colormap
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]

# We create an image with RGB colorized heatmap
jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

jet_heatmap_img = keras.preprocessing.image.array_to_img(jet_heatmap)
jet_heatmap_img.save("jet_heatmap_img.bmp")
# Superimpose the heatmap on original image
superimposed_img = jet_heatmap * 0.4 + img
superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

# Save the superimposed image
source_path = "source.bmp"
save_path = "superimposed_img.bmp"
img = keras.preprocessing.image.array_to_img(img)

superimposed_img.save(save_path)
img.save((source_path))
# Display Grad CAM
#display(Image(save_path))