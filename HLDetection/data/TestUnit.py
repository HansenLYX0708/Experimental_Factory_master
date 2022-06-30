from coco import COCODataSet

def Test1(img_dir, anno_path, dataset):
    dataset = COCODataSet(img_dir, anno_path, dataset)
    dataset.load_roidb_and_cname2cid()
    print("hello")


if __name__ == '__main__':
    img_dir = "train"
    anno_path = "annotations/train.json"
    dataset_path = "C:/data/Slider_abs/_coco_format"
    Test1(img_dir, anno_path, dataset_path)