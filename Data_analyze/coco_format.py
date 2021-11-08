import pycocotools.coco as coco
import os
import numpy as np
import scipy as spy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd


class Coco_analyze():
    def __init__(self, annotation_file):
        self.annotation_file = annotation_file

    def Data_load(self):
        self.coData = coco.COCO(self.annotation_file)

    def Get_all_cats(self):
        # Get categories to dict
        # cats = cocoanalyze.Get_all_cat()
        # Get categories to list
        self.cats = cocoanalyze.coData.loadCats(cocoanalyze.coData.getCatIds())
        return self.cats

    def Get_image_num_summary(self):
        self.Get_all_cats()
        summary = []
        self.cat_nms = [cat['name'] for cat in self.cats]
        for cat_name in self.cat_nms:
            catId = self.coData.getCatIds(catNms=[cat_name])
            imgId = self.coData.getImgIds(catIds=catId)
            annId = self.coData.getAnnIds(imgIds=imgId, catIds=catId, iscrowd=None)
            summary.append([cat_name, len(imgId), len(annId)])

        df = pd.DataFrame(summary, columns=['Name', 'Number of images', 'Number of labels'])

        return df


if __name__ == '__main__':
    annotation_file_1 = "C:\\data\\Slider_abs\\_coco_format\\annotations\\train.json"
    annotation_file_2 = "C:\\data\\xinye\\b_annotations.json"

    cocoanalyze = Coco_analyze(annotation_file_1)
    cocoanalyze.Data_load()
    df = cocoanalyze.Get_image_num_summary()
    df.plot(kind='bar')

    #plt.rcParams['figure.figsize'] = (8.0, 4.0)
    #plt.rcParams['savefig.dpi'] = 500
    #plt.rcParams['figure.dpi'] = 500
    plt.savefig("train.png")



    print('end')