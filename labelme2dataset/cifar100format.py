import os
import numpy as np


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

dict_label_to_id = {
    '0':0,
    '1':1,
    '2':2,
    '3':3,
    '4':4,
    '5':5,
    '6':6,
    '7':7,
    '8':8,
    '9':9,
    'A':10,
    'B':11,
    'C':12,
    'D':13,
    'E':14,
    'F':15,
}

def GetFilesName(root_path, main_folder, save_path):

    folders = os.listdir(os.path.join(root_path, main_folder))
    for folder in folders:
        print(folder)
        files = os.listdir(os.path.join(root_path,main_folder, folder))
        for file in files:
            with open(save_path, mode='a+', encoding='utf-8') as f:
                f.write(main_folder+ '/' + folder + '/' + file +' ' + str(dict_label_to_id[folder]) + '\n')


    return 0

if __name__ == '__main__':
    root_path = "C:\\data\\SliderSN"
    main_folder = "train"
    save_path = "C:\\data\\SliderSN\\train.txt"
    GetFilesName(root_path, main_folder, save_path)

    root_path = "C:\\data\\SliderSN"
    main_folder = "test"
    save_path = "C:\\data\\SliderSN\\test.txt"
    GetFilesName(root_path, main_folder, save_path)
