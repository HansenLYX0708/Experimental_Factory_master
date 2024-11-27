import os

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
train_txt_file = "train_list.txt"
val_txt_file = "val_list.txt"


if __name__ == '__main__':
    dir_list = os.listdir(data_path)

    for cat_name in dir_list:
        imglist = os.listdir(os.path.join(data_path, cat_name))
        train_num = int(len(imglist) * rate)
        for i in range(train_num):
            label = "pic/" + imglist[i] + " " + str(cat2id_dict[cat_name]) + "\n"
            with open(train_txt_file, 'a') as f:
                f.write(label)
        for i in range(train_num, len(imglist)):
            label = "pic/" + imglist[i] + " " + str(cat2id_dict[cat_name]) + "\n"
            with open(val_txt_file, 'a') as f:
                f.write(label)
        print(cat_name + ' end')


    print('end')


