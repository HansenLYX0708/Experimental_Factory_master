import os
from tensorflow import keras

datasets_path = "C:/data"

path_to_zip = keras.utils.get_file('facades.tar.gz',
                                  cache_subdir=datasets_path,#os.path.abspath('.'),
                                  origin='https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz',
                                  extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'facades/')
print('dataset path:', PATH)

