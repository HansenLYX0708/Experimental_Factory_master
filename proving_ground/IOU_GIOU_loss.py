import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np

from detection.core.loss.losses import (
bboxes_iou,
bboxes_giou,
bboxes_diou,
bboxes_ciou
)

if __name__ == '__main__':
    boxes1 = np.asarray([[0, 0, 5, 5], [0, 0, 10, 10], [3, 3, 10, 10]])
    boxes2 = np.asarray([[0, 0, 5, 5]])
    iou = bboxes_iou(boxes1, boxes2)
    print('end')