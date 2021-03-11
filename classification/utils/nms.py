import numpy as np


def nms(dets, Nt):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    order = scores.argsort()[::-1]
    # 计算面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 保留最后需要保留的边框的索引
    keep = []
    while order.size > 0:
        # order[0]是目前置信度最大的，肯定保留
        i = order[0]
        keep.append(i)

        # 计算窗口i与其他窗口的交叠的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算相交框的面积,不相交时用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 计算IOU：相交的面积/相并的面积
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr < thresh)[0]
        order = order[inds + 1]

    return keep


def py_cpu_softnms(dets, Nt=0.3, sigma=0.5, thresh=0.5, method=2):
    """
    py_cpu_softnms
    :param dets:   boexs 坐标矩阵 format [x1, y1, x2, y2, score]
    :param Nt:     iou 交叠阈值
    :param sigma:  使用 gaussian 函数的方差
    :param thresh: 最后的分数阈值
    :param method: 使用的方法，1：线性惩罚；2：高斯惩罚；3：原始 NMS
    :return:       留下的 boxes 的 index
    """

    N = dets.shape[0]
    # the order of boxes coordinate is [x1,y1,x2,y2]
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tB = dets[i, :4]
        ts = dets[i, 4]
        ta = areas[i]
        pos = i + 1

        if i != N - 1:
            maxscore = np.max(dets[:, 4][pos:])
            maxpos = np.argmax(dets[:, 4][pos:])
        else:
            maxscore = dets[:, 4][-1]
            maxpos = -1

        if ts < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :4] = tB

            dets[:, 4][i] = dets[:, 4][maxpos + i + 1]
            dets[:, 4][maxpos + i + 1] = ts

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = ta

        # IoU calculate
        xx1 = np.maximum(dets[i, 0], dets[pos:, 0])
        yy1 = np.maximum(dets[i, 1], dets[pos:, 1])
        xx2 = np.minimum(dets[i, 2], dets[pos:, 2])
        yy2 = np.minimum(dets[i, 3], dets[pos:, 3])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = weight[ovr > Nt] - ovr[ovr > Nt]
        elif method == 2:  # gaussian
            weight = np.exp(-(ovr * ovr) / sigma)
        else:  # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = 0

        dets[:, 4][pos:] = weight * dets[:, 4][pos:]

    # select the boxes and keep the corresponding indexes
    inds = np.argwhere(dets[:, 4] > thresh)
    keep = inds.astype(int).T[0]

    return keep


# test
if __name__ == "__main__":
    dets = np.array([[30, 20, 230, 200, 1],
                     [50, 50, 260, 220, 0.9],
                     [210, 30, 420, 5, 0.8],
                     [430, 280, 460, 360, 0.7]])
    thresh = 0.35
    keep_dets = nms(dets, thresh)
    print(keep_dets)
    print(dets[keep_dets])
