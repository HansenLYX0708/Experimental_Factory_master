import matplotlib.pyplot as plt
import numpy as np
import gzip

def CondelaImgFromSZ(input_filename, out_bin_filename, out_txt_filename):
    dtype = np.dtype('byte')
    fid = open(input_filename, 'rb')
    fid.seek(1059)
    data = np.fromfile(fid, dtype)
    tmp = data[:100]
    data2 = gzip.decompress(data)
    with open(out_bin_filename, 'wb') as f:
        f.write(data2)
    fid_bin = open(out_bin_filename, 'rb')
    data3 = np.fromfile(fid_bin, np.dtype("i2"))
    shape = (7789, 16384)
    convert = 0.00305185096
    data3 = data3 * convert
    image = data3.reshape(shape)
    np.savetxt(out_txt_filename, image, fmt='%f', delimiter=',')


def CondelaImgFromTHO(input_filename, out_bin_filename, out_txt_filename):
    dtype = np.dtype('byte')
    fid = open(input_filename, 'rb')
    data = np.fromfile(fid, dtype)
    tmp = data[:100]
    data2 = gzip.decompress(data)
    tmp2 = data2[:1141]
    data2 = data2[1131:]
    with open(out_bin_filename, 'wb') as f:
        f.write(data2)
    fid_bin = open(out_bin_filename, 'rb')
    data3 = np.fromfile(fid_bin, np.dtype("i2"))
    shape = (8326, 16384)
    convert = 0.00305185096
    data3 = data3 * convert
    image = data3.reshape(shape)
    np.savetxt(out_txt_filename, image, fmt='%f', delimiter=',')

if __name__ == '__main__':
    isSZ = False
    input_filename = "C:\\Users\\1000250081\\_work\\data\\Candela\\samples\\Test_PSc.img"
    out_bin_filename = "C:\\Users\\1000250081\\_work\\data\\Candela\\samples\\Test_PSc.bin"
    out_txt_filename = "C:\\Users\\1000250081\\_work\\data\\Candela\\samples\\Test_PSc.txt"
    # SZ
    # Wash 97mm Brush_Finger_Damage_6120O_19B_PSc.img    1002    (7789, 16384)
    # Wash 97mm Brush_Finger_Damage_6120O_19B_PScR.img    1059    (7789, 16384)

    # THO
    # Test_PSc.img    1131    (8326, 16384)

    if isSZ:
        CondelaImgFromSZ(input_filename, out_bin_filename, out_txt_filename)
    else:
        CondelaImgFromTHO(input_filename, out_bin_filename, out_txt_filename)
