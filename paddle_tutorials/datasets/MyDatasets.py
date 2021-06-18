import gzip
import struct
import numpy as np
import paddle
from paddle.io import Dataset
from PIL import Image
from paddle.vision.transforms import transforms


mnist = paddle.vision.datasets.MNIST()

class ImgTransforms(object):
    """
    并对图像的维度进行转换从HWC变为CHW
    """

    def __init__(self, fmt, dtype):
        self.format = fmt
        self.dtype = dtype

    def __call__(self, img):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        img = img.transpose(self.format)
        img = img.astype(self.dtype)
        img = img / 255.0
        #if img.shape[0] == 1:
        #    img = np.repeat(img, 3, axis=0)
        return img


class SliderSNDataset(Dataset):
    def __init__(self,
                 image_path,
                 label_path,
                 mode='train',
                 backend=None,
                 val_split=0.2
                 ):
        super(SliderSNDataset, self).__init__()
        self.mode = mode.lower()
        self.image_path = image_path
        self.label_path = label_path

        self.imgs = np.load(image_path)
        self.labels = np.load(label_path)

        if backend is None:
            backend = paddle.vision.get_image_backend()
        if backend not in ['pil', 'cv2']:
            raise ValueError(
                "Expected backend are one of ['pil', 'cv2'], but got {}"
                .format(backend))
        self.backend = backend
        self.dtype = paddle.get_default_dtype()

        # clean data if need

        # split train & val datasets
        if self.mode in ['train', 'val']:
            np.random.seed(43)
            data_len = len(self.imgs)
            shuffled_indices = np.random.permutation(data_len)
            # shuffled_indices = np.arange(data_len)
            self.shuffled_indices = shuffled_indices
            val_set_size = int(data_len * val_split)
            if self.mode == 'val':
                val_indices = shuffled_indices[:val_set_size]
                self.data_img = self.imgs[val_indices]
                self.data_label = self.labels[val_indices]
            elif self.mode == 'train':
                train_indices = shuffled_indices[val_set_size:]
                self.data_img = self.imgs[train_indices]
                self.data_label = self.labels[train_indices]
        elif self.mode == 'test':
            self.data_img = self.imgs
            self.data_label = self.labels

        # transform
        self.transform = transforms.Compose([
            ImgTransforms((2, 0, 1), self.dtype)
        ])


    def __getitem__(self, item):
        image, label = self.data_img[item], self.data_label[item]
        image = self.transform(image)
        image = paddle.to_tensor(image)
        label = paddle.to_tensor(label)
        return image, label

    def __len__(self):
        return len(self.data_img)


if __name__ == '__main__':
    img_path = 'C:/data/SliderSN/SliderSN_mnist/sn_x_train.npy'
    label_path = 'C:/data/SliderSN/SliderSN_mnist/sn_y_train.npy'
    mydataset = SliderSNDataset(img_path, label_path)