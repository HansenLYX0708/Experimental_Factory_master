import numpy as np
import math
import tensorflow as tf
import imageio

def psnr(target, ref, scale):
    # target:目标图像  ref:参考图像  scale:尺寸大小
    # assume RGB image
    target_data = np.array(target)
    target_data = target_data[scale:-scale, scale:-scale]

    ref_data = np.array(ref)
    ref_data = ref_data[scale:-scale, scale:-scale]

    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(1.0 / rmse)


def read_img(path):
    im = imageio.imread(path, as_gray=False, pilmode = "RGB")
    im = tf.image.convert_image_dtype(im, dtype=tf.float32)
    return im


def psnr(tf_img1, tf_img2):
    ret = tf.image.psnr(tf_img1, tf_img2, max_val=1.0)
    return ret

if __name__ == '__main__':
    folder = "C:/data/SliderSN_inference/"
    t1 = read_img(folder + "ocr_10_output_7_0.bmp")
    t2 = read_img(folder + "ocr_10_output_7_2_0.bmp")
    result = psnr(t1, t2)
    print(result.numpy())
    print('end')