import cv2


def guideFilter(I, p, winSize, eps):

    # I的均值平滑
    mean_I = cv2.blur(I, winSize)

    # p的均值平滑
    mean_p = cv2.blur(p, winSize)

    # I*I和I*p的均值平滑
    mean_II = cv2.blur( I *I, winSize)

    mean_Ip = cv2.blur( I *p, winSize)

    # 方差
    var_I = mean_II - mean_I * mean_I  # 方差公式

    # 协方差
    cov_Ip = mean_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a* mean_I

    # 对a、b进行均值平滑
    mean_a = cv2.blur(a, winSize)
    mean_b = cv2.blur(b, winSize)

    q = mean_a * I + mean_b

    return q


if __name__ == '__main__':
    img_source = "C:\\Users\\1000250081\\Desktop\\New folder (3)\\39726483 W3C (At critical area)\\1r.bmp"
    img = cv2.imread(img_source)
    pro = guideFilter(img, 1, (8,8), 0.01)
    cv2.imshow("guide", pro)

