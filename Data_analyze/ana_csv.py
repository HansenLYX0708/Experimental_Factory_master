import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def violin_plot(data, names):
    plt.figure(2, figsize=(32, 16))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, k in enumerate(names):
        plt.subplot(4, 4, i + 1)
        plt.title(k, fontsize=16)
        sns.violinplot(data=data[k], linewidth=2,
                       width=0.8, fliersize=3,
                       whis=1.5, notch=True,
                       scale='width', palette=[sns.color_palette("Set2", n_colors=14)[i]])

    plt.show()

if __name__ == '__main__':
    # file = "C:/Users/1000250081/_work/projects/ADCPeojrct/data/sdet vs quasi vs hdd_202209.csv"
    file = "C:/Users/1000250081/_work/projects/ADCPeojrct/data/ACC Model Data/LDS CMR/lds_cmr_30features.csv"

    df = pd.read_csv(file)
    print(df.info())
    # df.describe().to_csv("describe.csv")

    sns.pairplot(df)
    # data = df.to_list()

    # violin_plot(df.to_list, df.columns.to_list())

    print("end")
