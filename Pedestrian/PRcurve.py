# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from itertools import cycle
import scipy.io
# setup plot details

colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
lw = 2

result_names = ['','']
min_conf_threshold = 0.1 # 最小分类置信度阈值
conf_thresholds = np.linspace( min_conf_threshold, 1, 10 ) # 分类置信度阈值
# tps = np.array([17826, 17557, 16914, 16077, 15054, 13863, 12330, 10475, 8129, 5006, 2836], dtype=np.float64) # 正检
# fps = np.array([65275, 39675, 22468, 12002, 6273, 3662, 2268, 1442, 829, 305, 111], dtype=np.float64) # 误检
# fns = np.array([1005, 1274, 1917, 2754, 3777, 4968, 6501, 8356, 10702, 13825, 15995], dtype=np.float64) # 漏检
#
# tps2 = np.array([17311, 17035, 16505, 15684, 14679, 13500, 12042, 10312, 8044, 4898, 2591], dtype=np.float64) # 正检
# fps2 = np.array([155538, 71986, 28405, 13080, 6929, 4165, 2669, 1694, 971, 363, 119], dtype=np.float64) # 误检
# fns2 = np.array([1520, 1796, 2326, 3147, 4158, 5331, 6789,  8519, 10793, 13933, 16240], dtype=np.float64) # 漏检
#
# precision = np.divide(tps, np.add(tps, fps))
# recall = np.divide(tps ,np.add(tps, fns))
#
# precision2 = np.divide(tps2, np.add(tps2, fps2))
# recall2 = np.divide(tps2 ,np.add(tps2, fns2))
#
# print 'thresholds ',conf_thresholds
# print 'recall' , recall
# print 'precision' , precision
#
# print 'recall2' , recall2
# print 'precision2' , precision2
#
# # Plot Precision-Recall curve
# plt.clf()
# plt.plot(recall, precision, lw=lw, color='navy',
#          label='Precision-Recall curve')
# plt.plot(recall,precision,'ro')
# plt.plot(recall2, precision2, lw=lw, color='Orange',
#          label='Precision-Recall2 curve')
# plt.plot(recall2,precision2,'ro')
# #画对角线
# plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
#
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.ylim([0.0, 1.05])
# plt.xlim([0.0, 1.0])
# plt.title('Precision-Recall')
# plt.legend(loc="lower left")
# plt.grid()
# plt.savefig('PRcurve.png')
# plt.show()


def save_data(priorList, resultList, recall_mat):
    with open(priorList) as fp1, open(resultList) as fp2:  # 对于每个测试图片
        for priorFile in fp1:  # 每一行匹配数据 resultFile
            resultFile = fp2.readline()  # 每一行检测数据 priorFile
            prior_datas = priorFile.strip().split('\t')
            result_datas = resultFile.strip().split('\t')
            prior_boxes_total = int(prior_datas[2])  # 匹配box数量
            for i in range(0, prior_boxes_total, 1): # 对每个prior box
                conf = float(result_datas[6 * i + 2])  # 分类置信度
                for k in range(0, len(conf_thresholds), 1):  # 对于每个分类置信阈值
                    if conf >= conf_thresholds[k]:
                        all_change_group[k]['Pos'] += 1 # 正检
                    else:
                        all_change_group[k]['Neg'] += 1 # 漏检
    scipy.io.savemat(recall_mat,{ 'all_change_group': all_change_group})

def draw_curve(recall_num, data_mat_1, data_mat_2 = 0):
    fig, axes = plt.subplots(nrows=1, figsize=(8, 8))
    if recall_num == 1:
        data = scipy.io.loadmat(data_mat_1)
        data = data['all_change_group'][0]
        recalls = []
        precision = []
        for conf_i in range(0, len(conf_thresholds), 1):
            TP = float(data[conf_i]['Pos'])
            FP = float(data[conf_i]['Neg'])
            if TP == 0:
                recall = 0
            else:
                recall = TP / (TP + FP)
            recalls.append(recall)
        axes.plot(s_ids, recalls, lw=2, color=[colors[i] for i in recall_num],
                     label=str(conf_thresholds[conf_i]))  # 绘制每一条recall曲线


all_change_group = []  # 初始化
for j in range(0, len(conf_thresholds), 1):
    all_change_group.append({'Neg': 0, 'Pos': 0})
s_ids = np.arange(len(conf_thresholds))

if __name__ == "__main__":
    # save_data("./Data_0810/IOU_ALL_image_List.txt",
    #           "./Data_0810/result_ALL_image_List.txt",
    #           "./Data_0810/recall_1.mat")
    draw_curve(1,"./Data_0810/recall_1.mat",
               "")