import cv2
import numpy as np

def compare_img_default(img1, img2):
    """
    Strictly compare whether two pictures are equal
        Attention: Even just a little tiny bit different (like 1px dot), will return false.
    :param img1: img1 in MAT format(img1 = cv2.imread(image1))
    :param img2: img2 in MAT format(img2 = cv2.imread(image2))
    :return: true for equal or false for not equal
    """
    difference = cv2.subtract(img1, img2)
    result = not np.any(difference)
    return result


def compare_img_hist(img1, img2):
    """
    Compare the similarity of two pictures using histogram(直方图)
        Attention: this is a comparision of similarity, using histogram to calculate
        For example:
         1. img1 and img2 are both 720P .PNG file,
            and if compare with img1, img2 only add a black dot(about 9*9px),
            the result will be 0.999999999953
    :param img1: img1 in MAT format(img1 = cv2.imread(image1))
    :param img2: img2 in MAT format(img2 = cv2.imread(image2))
    :return: the similarity of two pictures
    """
    # Get the histogram data of image 1, then using normalize the picture for better compare
    img1_hist = cv2.calcHist([img1], [1], None, [256], [0, 256])
    img1_hist = cv2.normalize(img1_hist, img1_hist, 0, 1, cv2.NORM_MINMAX, -1)

    img2_hist = cv2.calcHist([img2], [1], None, [256], [0, 256])
    img2_hist = cv2.normalize(img2_hist, img2_hist, 0, 1, cv2.NORM_MINMAX, -1)
    similarity = cv2.compareHist(img1_hist, img2_hist, 0)
    return similarity


def compare_img_p_hash(img1, img2):
    """
    Get the similarity of two pictures via pHash
        Generally, when:
            ham_dist == 0 -> particularly like
            ham_dist < 5  -> very like
            ham_dist > 10 -> different image
        Attention: this is not accurate compare_img_hist() method, so use hist() method to auxiliary comparision.
            This method is always used for graphical search applications, such as Google Image(Use photo to search photo)
    :param img1:
    :param img2:
    :return:
    """
    hash_img1 = get_img_p_hash(img1)
    hash_img2 = get_img_p_hash(img2)

    return ham_dist(hash_img1, hash_img2)

def get_img_p_hash(img):
    """
    Get the pHash value of the image, pHash : Perceptual hash algorithm(感知哈希算法)
    :param img: img in MAT format(img = cv2.imread(image))
    :return: pHash value
    """
    hash_len = 32

    # GET Gray image
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Resize image, use the different way to get the best result
    resize_gray_img = cv2.resize(gray_img, (hash_len, hash_len), cv2.INTER_AREA)
    # resize_gray_img = cv.resize(gray_img, (hash_len, hash_len), cv.INTER_LANCZOS4)
    # resize_gray_img = cv.resize(gray_img, (hash_len, hash_len), cv.INTER_LINEAR)
    # resize_gray_img = cv.resize(gray_img, (hash_len, hash_len), cv.INTER_NEAREST)
    # resize_gray_img = cv.resize(gray_img, (hash_len, hash_len), cv.INTER_CUBIC)
    # Change the int of image to float, for better DCT
    h, w = resize_gray_img.shape[:2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = resize_gray_img
    # DCT: Discrete cosine transform(离散余弦变换)
    vis1 = cv2.dct(cv2.dct(vis0))
    vis1.resize(hash_len, hash_len)
    img_list = vis1.flatten()
    # Calculate the avg value
    avg = sum(img_list) * 1. / len(img_list)
    avg_list = []
    for i in img_list:
        if i < avg:
           tmp = '0'
        else:
            tmp = '1'
        avg_list.append(tmp)
    # Calculate the hash value
    p_hash_str = ''
    for x in range(0, hash_len * hash_len, 4):
        p_hash_str += '%x' % int(''.join(avg_list[x:x + 4]), 2)
    return p_hash_str

def ham_dist(x, y):
    """
    Get the hamming distance of two values.
        hamming distance(汉明距)
    :param x:
    :param y:
    :return: the hamming distance
    """
    assert len(x) == len(y)
    return sum([ch1 != ch2 for ch1, ch2 in zip(x, y)])


if __name__ == '__main__':
    img1 = cv2.imread("C:/data/slider_abs_6/12_ROI/1622190644356.bmp")
    img2 = cv2.imread("C:/data/slider_abs_6/12_ROI/1622190678418.bmp")

    # func1 = compare_img_default(img1, img2)
    # func2 = compare_img_hist(img1, img2)
    c = compare_img_p_hash(img1, img2)
    print(c)