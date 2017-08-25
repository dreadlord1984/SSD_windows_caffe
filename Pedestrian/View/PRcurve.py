# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import scipy.io
import xml.etree.cElementTree as et
# setup plot details


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

"""
@function:计算两个box的IOU
@param param1: box1
@param param2: box2
@return: IOU
"""
def computIOU(A, B):
    W = min(A[2], B[2]) - max(A[0], B[0])
    H = min(A[3], B[3]) - max(A[1], B[1])
    if (W <= 0 or H <= 0):
        return 0
    SA = (A[2] - A[0]) * (A[3] - A[1])
    SB = (B[2] - B[0]) * (B[3] - B[1])
    cross = W * H
    iou = float(cross) / (SA + SB - cross)
    return iou

"""
@function:从xml文件中读取box信息
@param param1: xml文件
@return: boxes
"""
def readXML(xml_name):
    tree = et.parse(xml_name) #打开xml文档
    # 得到文档元素对象
    root = tree.getroot()
    size = root.find('size')  # 找到root节点下的size节点
    width = int(size.find('width').text)  # 子节点下节点width的值
    height = int(size.find('height').text)  # 子节点下节点height的值

    boundingBox = []
    for object in root.findall('object'):  # 找到root节点下的所有object节点
        bndbox = object.find('bndbox')  # 子节点下属性bndbox的值
        xmin = bndbox.find('xmin').text
        ymin = bndbox.find('ymin').text
        xmax = bndbox.find('xmax').text
        ymax = bndbox.find('ymax').text
        boundingBox.append([int(xmin), int(ymin), int(xmax), int(ymax)])
    return boundingBox, width, height

"""
@function:将不同conf阈值下的TP、FP、FN结果保存
@param param1: 测试列表文件
@param param2: 检测结果列表文件
@param param3: 待保存mat文件
"""
def save_data(testList, resultList, recall_mat):
    with open(testList) as fp1, open(resultList) as fp2:  # 对于每个测试图片
        for testFile in fp1:  # 每一行匹配数据 resultFile
            resultFile = fp2.readline()  # 每一行检测数据 priorFile
            img_name = ROOTDIR + testFile.strip().split('.jpg ')[0]
            xml_name = ROOTDIR + testFile.strip().split('.jpg ')[1]
            true_boxes, width, height = readXML(xml_name)  # 所有的ground truth boxes

            result_datas = resultFile.strip().split('\t')
            result_boxes_total = int(result_datas[1])  # 匹配box数量
            result_boxes = []
            for i in range(0, result_boxes_total, 1): # 对每个result box
                conf = float(result_datas[5 * i + 2])  # 分类置信度
                xmin = int(result_datas[5 * i + 3])
                ymin = int(result_datas[5 * i + 4])
                xmax = int(result_datas[5 * i + 5])
                ymax = int(result_datas[5 * i + 6])
                result_boxes.append([conf,[xmin, ymin, xmax, ymax]])

            for conf_i in range(0, len(conf_thresholds), 1):  # 对于每个分类置信阈值
                TP = 0  # 正检
                FP = 0  # 误检
                FN = 0  # 漏检
                for result_box in result_boxes:  # 对每个result box
                    not_match = 0
                    if result_box[0] >= conf_thresholds[conf_i]: # 属于该分类阈值下的检测结果
                       for boxT in true_boxes:
                           if (computIOU(boxT, result_box[1]) < 0.5):
                               not_match += 1  # 未匹配次数
                       if not_match == len(true_boxes): # 没有一个gt box能和result box匹配则为误检
                           FP += 1
                for boxT in true_boxes:
                    for result_box in result_boxes:
                        if result_box[0] >= conf_thresholds[conf_i]:  # 属于该分类阈值下的检测结果
                            if (computIOU(boxT, result_box[1]) > 0.5):  # 如果有任意一个检测框能和ground_truth_box 匹配上则TP+1
                                TP += 1  # 正确检测
                                break
                FN = len(true_boxes) - TP
                all_change_group[conf_i]['TP'] += TP
                all_change_group[conf_i]['FP'] += FP
                all_change_group[conf_i]['FN'] += FN

    scipy.io.savemat(recall_mat,{ 'all_change_group': all_change_group})

"""
@function:绘制PR曲线
@param param1: 模型结果数量，模型一结果，模型二结果
"""
def draw_curve(recall_num, data_mat_1, data_mat_2 = 0):
    fig, axes = plt.subplots(nrows=1, figsize=(8, 8))
    if recall_num == 1:
        data = scipy.io.loadmat(data_mat_1)
        data = data['all_change_group'][0]
        recalls = []
        precisions = []
        for conf_i in range(0, len(conf_thresholds), 1):
            TP = float(data[conf_i]['TP'])
            FP = float(data[conf_i]['FP'])
            FN = float(data[conf_i]['FN'])
            if TP == 0:
                recall = 0
                precision = 0
            else:
                recall = TP / (TP + FP)
                precision = TP / (TP + FN)
            recalls.append(recall)
            precisions.append(precision)
        axes.plot(recalls, precisions, lw=2, color='navy',
                     label=str(conf_thresholds[conf_i]))  # 绘制每一条recall曲线
        plt.plot(recalls, precisions, 'ro')
    else:
        for curve_i in range(0, recall_num, 1):
            data_name = 'recall_%s' %str(curve_i) + '.mat'
            data = scipy.io.loadmat(data_name)
            data = data['all_change_group'][0]
            recalls = []
            precisions = []
            for conf_i in range(0, len(conf_thresholds), 1):
                TP = float(data[conf_i]['TP'])
                FP = float(data[conf_i]['FP'])
                FN = float(data[conf_i]['FN'])
                if TP == 0:
                    recall = 0
                    precision = 0
                else:
                    recall = TP / (TP + FP)
                    precision = TP / (TP + FN)
                recalls.append(recall)
                precisions.append(precision)
            axes.plot(recalls, precisions, lw=2, color=colors[curve_i],
                      label=data_name )  # 绘制每一条recall曲线
            plt.plot(recalls, precisions, 'ro')
    plt.legend(loc="lower left")
    #画对角线
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall')
    plt.show()


colors = plt.cm.hsv(np.linspace(0, 1, 10)).tolist()
lw = 2
conf_thresholds = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95], dtype=np.float64)
ROOTDIR = "\\\\192.168.1.186/PedestrianData/"
all_change_group = []  # 初始化
for j in range(0, len(conf_thresholds), 1):
    all_change_group.append({'TP': 0, 'FP': 0, 'FN':0})
s_ids = np.arange(len(conf_thresholds))




if __name__ == "__main__":
    # save_data("val.txt",
    #           "result_conf_1.txt",
    #           "recall_1.mat")
    draw_curve(1, "recall_1.mat")