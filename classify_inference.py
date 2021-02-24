from classification.utils.tf_2_pb_to_frozen_graph import load_frozen_model_inference
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

dict_id_to_label = {
    0:'0',
    1:'1',
    2:'2',
    3:'3',
    4:'4',
    5:'5',
    6:'6',
    7:'7',
    8:'8',
    9:'9',
    10:'A',
    11:'B',
    12:'C',
    13:'D',
    14:'E',
    15:'F',
}

if __name__ == '__main__':
    frozen_folder = "G:\\github\\Experimental_Factory_master\\frozen_models"
    frozen_name = "frozen_model.pb"
    img_path = "G:\\_work\\_data\\sliderSN_data\\SliderSN_inference\\"
    inference_model = load_frozen_model_inference(os.path.join(frozen_folder, frozen_name))
    x = []
    imgs_list = os.listdir(img_path)
    for img_name in imgs_list:
        img = Image.open(os.path.join(img_path, img_name))
        img = np.array(img.convert('L'))
        img_input = img.astype(np.float32) / 255.0
        x.append(img_input)
    x = np.array(x)
    imgs_input = np.expand_dims(x, axis=3)
    logit = inference_model(tf.constant(imgs_input))
    pred = tf.argmax(logit, axis=2)
    preds_np = pred.numpy()[0]
    preds_label = []
    for label in preds_np:
        preds_label.append(dict_id_to_label[label])

    plt.figure()
    for i in range(1, 21):
        plt.subplot(4, 5, i)
        plt.imshow(x[i - 1], cmap ='gray')
        plt.title(preds_label[i - 1], fontsize=10, color='Blue',bbox=dict(facecolor="yellow", edgecolor='black', alpha=0.65))
        plt.xticks([])
        plt.yticks([])

    plt.show()

    print("end ")
