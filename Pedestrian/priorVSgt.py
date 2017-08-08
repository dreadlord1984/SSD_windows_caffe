# -*- coding: utf-8 -*-
import numpy as np
import pylab as plt
import matplotlib
import prettyplotlib as ppl

thresholds = np.linspace( 0.1, 1, 10 ) # 分段阈值
s_ids = np.arange(thresholds.size)
group = np.zeros(thresholds.size,dtype=np.int32)
def draw_curve(dataList):
    for data in open(dataList).readlines():  # 对于每个数据
        IOU = float(data.strip().split(' ')[1])
        for i in s_ids:
            if (IOU < thresholds[i]):
                group[i] += 1
                break
    matplotlib.rcParams['figure.figsize'] = (8, 5)  # 设定显示大小
    fig, ax = plt.subplots(1)
    labels = [thresholds[i] for i in s_ids]
    anno_area2s = [('%d' % a) for a in group[s_ids]]
    ppl.bar(ax, np.arange(len(group)), group[s_ids], annotate=anno_area2s, grid='y', xticklabels=labels)
    plt.xticks(rotation=25)
    ax.set_title('(prior vs ground_truth)')
    ax.set_ylabel('Distribution')
    plt.savefig('Data_0728/bar.png')
    plt.show()

if __name__ == "__main__":
    draw_curve("Data_0728/noCropeALL.txt")

