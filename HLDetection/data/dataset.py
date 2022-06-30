import os

class DataSet(object):
    """
    Dataset, e.g., coco, pascal voc

    Args:
        annotation (str): annotation file path
        image_dir (str): directory where image files are stored
        shuffle (bool): shuffle samples
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 sample_num=-1,
                 with_background=True,
                 use_default_label=False,
                 **kwargs):
        super(DataSet, self).__init__()
        self.anno_path = anno_path
        self.image_dir = image_dir if image_dir is not None else ''
        self.dataset_dir = dataset_dir if dataset_dir is not None else ''
        self.sample_num = sample_num
        self.with_background = with_background
        self.use_default_label = use_default_label

        self.cname2cid = None
        self._imid2path = None

    def load_roidb_and_cname2cid(self):
        """load dataset"""
        raise NotImplementedError('%s.load_roidb_and_cname2cid not available' %
                                  (self.__class__.__name__))

    def get_roidb(self):
        if not self.roidbs:
            # data_dir = get_dataset_path(self.dataset_dir, self.anno_path,
            #                            self.image_dir)
            # if data_dir:
            #    self.dataset_dir = data_dir
            self.load_roidb_and_cname2cid()
        return self.roidbs

    def get_cname2cid(self):
        if not self.cname2cid:
            self.load_roidb_and_cname2cid()
        return self.cname2cid

    def get_anno(self):
        if self.anno_path is None:
            return
        return os.path.join(self.dataset_dir, self.anno_path)

    def get_imid2path(self):
        return self._imid2path
