import numpy as np
from PIL import Image
from paddle.io import Dataset


class MyDataset(Dataset):

    def __init__(self, image, label, transform=None):
        super(MyDataset, self).__init__()
        imgs = image
        labels = label

        self.labels = labels
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        im, la = self.imgs[index], self.labels[index]
        img = Image.open(im)
        img = img.convert("RGB")
        img = np.array(img)
        label = np.array([la]).astype(dtype='int64')

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
