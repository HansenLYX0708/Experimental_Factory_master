
from sharpness_metrics import *
import array as arr

def grab_roi(img_path, save_path, y1, x1, y2, x2):
    img = cv2.imread(img_path)
    cv2.imwrite(save_path, img[x1:x2, y1:y2])

def calculate_sharpness(img_path, y1, x1, y2, x2):
    img = cv2.imread(img_path)
    roi = img[x1:x2, y1:y2]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    #brenner_value = brenner(roi_gray)
    #Laplacian_value = Laplacian(roi_gray)
    #SMD_value = SMD(roi_gray)
    #SMD2_value = SMD2(roi_gray)
    #variance_value = variance(roi_gray)
    Tenengrad_value = Tenengrad(roi_gray)

    #print(brenner_value)
    #print(Laplacian_value)
    #print(SMD_value)
    #print(SMD2_value)
    #print(variance_value)
    #print(Tenengrad_value)
    return Tenengrad_value

def roi_imgs():
    '''
    function test
    :return:
    '''
    imgs_folder = "C:\\Users\\1000250081\\_work\\data\\rowbar\\depo inspection\\pad_to_base_4x"
    save_folder = "C:\\Users\\1000250081\\_work\\data\\rowbar\\depo inspection\\pad_to_base_4x_roi"

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    img_list = os.listdir(imgs_folder)
    for img_ in img_list:
        img_path = os.path.join(imgs_folder, img_)
        save_path = os.path.join(save_folder, img_)
        grab_roi(img_path, save_path, 2906, 2941, 3224, 3324)

def calculate_s():
    imgs_folder = "C:\\Users\\1000250081\\_work\data\\rowbar\\STEP-CAPTURE-4X"
    # save_folder = "C:\\Users\\1000250081\\_work\\data\\rowbar\\depo inspection\\pad_to_base_4x_roi_roi6"
    save_folder = "C:\\Users\\1000250081\\_work\data\\rowbar\\STEP-CAPTURE-4X_pad"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # sharpness_list_area1 = []
    sharpness_list_area1_pad = arr.array('f', [])
    sharpness_list_area1_plus = arr.array('f', [])
    sharpness_list_area1_olg = arr.array('f', [])
    sharpness_list_area1_sn = arr.array('f', [])
    sharpness_list_area1_olg2 = arr.array('f', [])
    sharpness_list_area1_base = arr.array('f', [])
    sharpness_list_area1_base1 = arr.array('f', [])
    img_list = os.listdir(imgs_folder)
    for img_ in img_list:
        img_path = os.path.join(imgs_folder, img_)
        save_path = os.path.join(save_folder, img_)
        # 120, 84, 147, 109
        # 156, 36, 255, 103
        # 39, 252, 138, 375
        # 254, 151, 315, 211
        # 38, 108, 71, 245
        # 117, 156, 198, 206
        # grab_roi(img_path, save_path, 1327,1403,1417,1539)
        ''' 6X
        sharpness_list_area1_pad.append(calculate_sharpness(img_path, 1327,1403,1417,1539))
        sharpness_list_area1_plus.append(calculate_sharpness(img_path, 1390,1344,1432,1384))
        sharpness_list_area1_olg.append(calculate_sharpness(img_path, 1500,1580,1584,1651))
        sharpness_list_area1_sn.append(calculate_sharpness(img_path, 1654,1291,1874,1375))
        sharpness_list_area1_olg2.append(calculate_sharpness(img_path, 1502,1340,1577,1452))
        sharpness_list_area1_base.append(calculate_sharpness(img_path, 1432,1221,1637,1270))
        sharpness_list_area1_base1.append(calculate_sharpness(img_path, 477,1187,708,1310))
        1346,1448,1391,1490
        '''
        #sharpness_list_area1_pad.append(calculate_sharpness(img_path, 1180,1183,1341,1266))
        grab_roi(img_path, save_path, 1862,1448,1918,1494)
        ''' 4x
        sharpness_list_area1_pad.append(calculate_sharpness(img_path, 1746,1330,1806,1420))
        sharpness_list_area1_plus.append(calculate_sharpness(img_path, 1788,1292,1817,1316))
        sharpness_list_area1_olg.append(calculate_sharpness(img_path, 1862,1448,1918,1494))
        sharpness_list_area1_sn.append(calculate_sharpness(img_path, 1964,1208,2106,1309))
        sharpness_list_area1_olg2.append(calculate_sharpness(img_path, 1863,1285,1914,1366))
        sharpness_list_area1_base.append(calculate_sharpness(img_path, 1811,1208,1962,1242))
        sharpness_list_area1_base1.append(calculate_sharpness(img_path, 1166,1185,1354,1267))
        '''
    print(sharpness_list_area1_pad)
    print(sharpness_list_area1_plus)
    print(sharpness_list_area1_olg)
    print(sharpness_list_area1_sn)
    print(sharpness_list_area1_olg2)
    print(sharpness_list_area1_base)
    print(sharpness_list_area1_base1)

if __name__ == '__main__':
    calculate_s()

