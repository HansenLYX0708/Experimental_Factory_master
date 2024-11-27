import os
import numpy as np
import pandas as pd
'''
5-fold dataset
same image have same name, so I change the name, generate "train_list_fix.txt"
'''


'''
0 Clean_ins
1 DATI
2 OffTrkWrite
3 Scratch_radial
4 Ser_deg_ins
5 Ser_instability
6 Spacing_MD_PW_SerDeg
7 Wr_related_1sct
'''


label_list_file = "C:/Users/1000250081/_work/projects/DataHackathon2022/BitflipImageClassification/code/PaddleClas/ppcls/utils/bitflip_label_list.txt"

data_path = "C:/Users/1000250081/_work/projects/DataHackathon2022/BitflipImageClassification/data/train_8c_crop_4comp/"

id2cat_dict = {0: 'Clean_ins', 1: 'DATI', 2: 'OffTrkWrite', 3: 'Scratch_radial',
               4: 'Ser_deg_ins', 5: 'Ser_instability', 6: 'Spacing_MD_PW_SerDeg', 7: 'Wr_related_1sct'}

cat2id_dict = {'Clean_ins': 0, 'DATI': 1, 'OffTrkWrite': 2, 'Scratch_radial': 3,
               'Ser_deg_ins': 4, 'Ser_instability': 5, 'Spacing_MD_PW_SerDeg': 6, 'Wr_related_1sct': 7}

rate = 0.8
train_fix_txt_file = "C:/Users/1000250081/_work/projects/DataHackathon2022/BitflipImageClassification/data/train_list_fix.txt"
val_fix_txt_file = "C:/Users/1000250081/_work/projects/DataHackathon2022/BitflipImageClassification/data/val_list_fix.txt"

if __name__ == '__main__':
    train_list = pd.read_csv(train_fix_txt_file, header=None, delimiter=" ")
    val_list = pd.read_csv(val_fix_txt_file, header=None,  delimiter=" ")

    data_all = pd.concat([train_list, val_list])
    #data_all_shuffle = data_all.sample(frac=1)
    data_all_shuffle = data_all.sample(frac=1)

    val_len = int(len(data_all_shuffle) * (1-rate))

    # 1st fold
    val_list_1fold = data_all_shuffle[:val_len]
    train_list_1fold = data_all_shuffle[val_len:]
    val_list_1fold.to_csv("val_list_1fold.txt", sep=" ", index=False, header=False)
    train_list_1fold.to_csv("train_list_1fold.txt", sep=" ", index=False, header=False)
    # 2nd fold
    val_list_2fold = data_all_shuffle[val_len : val_len * 2]
    train_list_2fold = pd.concat([data_all_shuffle[:val_len], data_all_shuffle[val_len * 2:]])
    val_list_2fold.to_csv("val_list_2fold.txt", sep=" ", index=False, header=False)
    train_list_2fold.to_csv("train_list_2fold.txt", sep=" ", index=False, header=False)
    # 3rd fold
    val_list_3fold = data_all_shuffle[val_len * 2: val_len * 3]
    train_list_3fold = pd.concat([data_all_shuffle[:val_len * 2], data_all_shuffle[val_len * 3:]])
    val_list_3fold.to_csv("val_list_3fold.txt", sep=" ", index=False, header=False)
    train_list_3fold.to_csv("train_list_3fold.txt", sep=" ", index=False, header=False)
    # 4th fold
    val_list_4fold = data_all_shuffle[val_len * 3: val_len * 4]
    train_list_4fold = pd.concat([data_all_shuffle[:val_len * 3], data_all_shuffle[val_len * 4:]])
    val_list_4fold.to_csv("val_list_4fold.txt", sep=" ", index=False, header=False)
    train_list_4fold.to_csv("train_list_4fold.txt", sep=" ", index=False, header=False)
    # 5th fold
    val_list_5fold = data_all_shuffle[len(data_all_shuffle) - val_len:]
    train_list_5fold = data_all_shuffle[:len(data_all_shuffle) - val_len]
    val_list_5fold.to_csv("val_list_5fold.txt", sep=" ", index=False, header=False)
    train_list_5fold.to_csv("train_list_5fold.txt", sep=" ", index=False, header=False)
    print('end')