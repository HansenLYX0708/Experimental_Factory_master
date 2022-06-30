from pycocotools.coco import COCO
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy


def cas_iou(box,cluster):
    x = np.minimum(cluster[:,0],box[0])
    y = np.minimum(cluster[:,1],box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cluster[:,0] * cluster[:,1]
    iou = intersection / (area1 + area2 -intersection)

    return iou

def kmeans2D(box, k):
    row = box.shape[0]
    distance = np.empty((row, k))
    last_clu = np.zeros((row,))
    np.random.seed()
    cluster = box[np.random.choice(row, k, replace=False)]
    while True:
        for i in range(row):
            distance[i] = 1 - cas_iou(box[i], cluster)
        near = np.argmin(distance, axis=1)
        if (last_clu == near).all():
            break

        for j in range(k):
            cluster[j] = np.median(
                box[near == j], axis=0)
        last_clu = near
    return cluster

def kmeans1D(values, k):
    row = values.shape[0]
    distance = np.empty((row, k))
    last_clu = np.zeros((row,))
    np.random.seed()
    cluster = values[np.random.choice(row, k, replace=False)]
    # cluster = random.sample(row, k)
    while True:
        for i in range(row):
            #distance[i] = 1 - cas_iou(values[i], cluster)
            distance[i] = abs(values[i] - cluster)
        near = np.argmin(distance, axis=1)
        if (last_clu == near).all():
            break

        for j in range(k):
            cluster[j] = np.median(
                values[near == j], axis=0)
        last_clu = near
    return cluster

def Get_data():
    abs_anns_path = "C:/data/slider_abs/_coco_format/annotations/instance_val.json"
    abs_anns = COCO(abs_anns_path)
    sample_ids = abs_anns.imgs.keys()
    samples_num = len(sample_ids)
    sample_classes = dict([(v["id"], v["name"]) for k, v in abs_anns.cats.items()])
    classes_count = dict([(v["id"], 0) for k, v in abs_anns.cats.items()])
    areas = []
    w_h_ratios = []
    boxs = []
    for sam_id in sample_ids:
        ann_ids = abs_anns.getAnnIds(imgIds=sam_id)

        targets = abs_anns.loadAnns(ann_ids)
        test = abs_anns.loadImgs(sam_id)
        for target in targets:
            areas.append(target['area'])
            w_h_ratios.append(target['bbox'][2] / target['bbox'][3])
            classes_count[target['category_id']] = classes_count[target['category_id']] + 1
            boxs.append([target['bbox'][2], target['bbox'][3]])

    areas = np.array(areas)

    return np.array(boxs) , areas

def test1():
    plt.subplot(221)
    plt.hist(areas)

    # obtain histogram data
    plt.subplot(222)
    hist, bin_edges = np.histogram(areas)
    plt.plot(hist)

    # fit histogram curve
    plt.subplot(223)
    sns.distplot(areas, kde=False, fit=scipy.stats.gamma, rug=True)
    plt.show()


if __name__ == '__main__':
    boxs, areas = Get_data()
    out = kmeans2D(boxs, 9)
    out_ratio = out.copy()
    out_ratio[:,0] = out_ratio[:,0] / 1280
    out_ratio[:,1] = out_ratio[:,1] / 1024

    out2 = kmeans1D(areas, 9)
    print(out)
    print(out_ratio)
    print(out2)
    # draw area cdf and pdf
    norm_cdf = scipy.stats.norm.pdf(areas)
    #sns.lineplot(x=areas, y=norm_cdf)
    #plt.show()
    len = np.zeros(11)
    len2 = np.zeros(11)

    for i in range(11):
        len[i] = sum(areas < (i * 20) * (i * 20))
        if i > 0:
            a = areas < (i * 20) * (i * 20)
            b = areas >= (i-1) * 20 * (i-1) * 20
            len2[i] = sum(a & b)
    len = len / 379
    height = len2 / 379

    fig, ax1 = plt.subplots()

    ax1.bar(x=range(-10,201, 20), height=height, color='#FFB6C1', width=10)
    ax1.set_ylabel("Probability Density")

    ax2 = ax1.twinx()
    ax2.plot(range(0, 201, 20), len)
    ax2.set_ylabel("Cumulative Distribution")

    x_ticks = np.arange(0, 220, 20)
    y_ticks = np.arange(0, 1.1, 0.1)

    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    plt.xlim((-1, 210))
    plt.savefig('instance_val.png')
    plt.show()


    print("end")
