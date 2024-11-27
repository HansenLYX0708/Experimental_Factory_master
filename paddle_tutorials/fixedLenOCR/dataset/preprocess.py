import os
import pandas as pd
import numpy as np

all_file_dir = 'C:\\data\\foods'

img_list = []
label_list = []

label_id = 0

class_list = [c for c in os.listdir(all_file_dir) if os.path.isdir(os.path.join(all_file_dir, c))]

for class_dir in class_list:
    img_path_pre = os.path.join(all_file_dir, class_dir)

    for img in os.listdir(img_path_pre):
        img_list.append(os.path.join(img_path_pre, img))
        label_list.append(label_id)
    label_id += 1

img_df = pd.DataFrame(img_list)
label_df = pd.DataFrame(label_list)
img_df.columns = ['images']
label_df.columns = ['label']
df = pd.concat([img_df, label_df], axis=1)

df = df.reindex(np.random.permutation(df.index))

df.to_csv('C:\\data\\foods\\food_data.csv')




