import os
import numpy as np
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

from dataset import DataSet


class ImageFolder(DataSet):
    """
    Args:
        dataset_dir (str): root directory for dataset.
        image_dir(list|str): list of image folders or list of image files
        anno_path (str): annotation file path.
        samples (int): number of samples to load, -1 means all
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 sample_num=-1,
                 with_background=True,
                 use_default_label=False,
                 **kwargs):
        super(ImageFolder, self).__init__(dataset_dir, image_dir, anno_path,
                                          sample_num, with_background,
                                          use_default_label)
        self.roidbs = None
        self._imid2path = {}

    def get_roidb(self):
        if not self.roidbs:
            self.roidbs = self._load_images()
        return self.roidbs

    def set_images(self, images):
        self.image_dir = images
        self.roidbs = self._load_images()

    def _parse(self):
        image_dir = self.image_dir
        if not isinstance(image_dir, Sequence):
            image_dir = [image_dir]
        images = []
        for im_dir in image_dir:
            if os.path.isdir(im_dir):
                im_dir = os.path.join(self.dataset_dir, im_dir)
                images.extend(_make_dataset(im_dir))
            elif os.path.isfile(im_dir) and _is_valid_file(im_dir):
                images.append(im_dir)
        return images

    def _load_images(self):
        images = self._parse()
        ct = 0
        records = []
        for image in images:
            assert image != '' and os.path.isfile(image), \
                    "Image {} not found".format(image)
            if self.sample_num > 0 and ct >= self.sample_num:
                break
            rec = {'im_id': np.array([ct]), 'im_file': image}
            self._imid2path[ct] = image
            ct += 1
            records.append(rec)
        assert len(records) > 0, "No image file found"
        return records

def _is_valid_file(f, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    return f.lower().endswith(extensions)


def _make_dataset(data_dir):
    data_dir = os.path.expanduser(data_dir)
    if not os.path.isdir(data_dir):
        raise ('{} should be a dir'.format(data_dir))
    images = []
    for root, _, fnames in sorted(os.walk(data_dir, followlinks=True)):
        for fname in sorted(fnames):
            file_path = os.path.join(root, fname)
            if _is_valid_file(file_path):
                images.append(file_path)
    return images