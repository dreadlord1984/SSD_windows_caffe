# -*- coding: utf-8 -*-
import numpy as np
import pylab as plt
import matplotlib
import prettyplotlib as ppl


"""
@function:绘制所有IOU区间段的匹配数量直方图
@param param1: 匹配列表
"""
def draw_curve(dataList):
    for data in open(dataList).readlines():  # 对于每个数据
        IOU = float(data.strip().split(' ')[1])
        for i in s_ids:
            if (IOU < thresholds[i]):
                group[i] += 1
                break

    total = np.sum(group)
    fig, ax = plt.subplots(1)
    labels = thresholds.tolist()
    # labels = [thresholds[i] for i in s_ids]
    anno_area2s = [('%d' % a) for a in group[s_ids]]
    ppl.bar(ax, np.arange(len(group)), group[s_ids], annotate=anno_area2s, grid='y', xticklabels=labels)
    plt.xticks(rotation=25)
    ax.set_title('(prior vs IOU, prior box: %d)' %total)
    ax.set_xlabel('IOU')
    ax.set_ylabel('prior boxes num')
    savename = dataList[:dataList.rfind("\\")] + "\\prior_IOU.png"
    plt.savefig(savename)
    plt.show()

thresholds = np.linspace( 0.1, 1, 10 ) # 进入训练的prior box匹配IOU分段阈值
s_ids = np.arange(thresholds.size)
group = np.zeros(thresholds.size,dtype=np.int32)
matplotlib.rcParams['figure.figsize'] = (8, 6)  # 设定显示大小

if __name__ == "__main__":
    draw_curve("..\\Data_0922\\IOU_ALL.txt")