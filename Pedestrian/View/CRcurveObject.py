# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import scipy.io
import xml.etree.cElementTree as et

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
@return: boxes(with area_ratio), width, height
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
        area_ratio = float((int(xmax) - int(xmin)) * (int(ymax) - int(ymin))) / (width * height)
        boundingBox.append([int(xmin), int(ymin), int(xmax), int(ymax), area_ratio])
    return boundingBox, width, height

"""
@function:将不同conf阈值下的TP、FP、FN结果保存
@param param1: 测试列表文件
@param param2: 检测结果列表文件(无后缀）
@output: 检测结果名+object.mat文件
"""
def save_data(testList, resultList):
    CR_mat = resultList.strip() + '_Object.mat'
    with open(testList) as fp1, open(resultList + '.txt') as fp2:  # 对于每个测试图片
        for testFile, resultFile in zip(fp1, fp2):
            # img_name = ROOTDIR + testFile.strip().split('.jpg ')[0]
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
                for boxT in true_boxes:
                    for area_i in range(0, len(area_thresholds), 1):  # 判断area区间段
                        if (boxT[4] <= area_thresholds[area_i]):
                            area_index = area_i
                            break

                    TP = 0  # 正检
                    for result_box in result_boxes:
                        if result_box[0] >= conf_thresholds[conf_i]:  # 属于该分类阈值下的检测结果
                            if (computIOU(boxT, result_box[1]) >= 0.5):  # 如果有任意一个检测框能和ground_truth_box 匹配上则TP+1
                                TP += 1  # 正确检测
                                break
                    FN = 1 if TP == 0 else 0 # 漏检
                    all_change_group[area_index][conf_i]['FN'] += FN
                    all_change_group[area_index][conf_i]['TP'] += TP

    scipy.io.savemat(CR_mat,{ 'all_change_group_Object': all_change_group})

"""
@function:绘制PR曲线
@param param1: 模型统计结果
"""
def draw_curve(data_mat):
    plt.rcParams['figure.figsize'] = (9, 10)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    data = scipy.io.loadmat(data_mat + '_Object.mat')
    data = data['all_change_group_Object']
    for area_i in range(0, len(area_thresholds), 1):  # 判断area区间段
        recalls = []
        for conf_i in range(0, len(conf_thresholds), 1):
            TP = float(data[area_i][conf_i]['TP'])
            FN = float(data[area_i][conf_i]['FN'])
            if TP == 0:
                recall = 0
            else:
                recall = TP / (TP + FN)
            recalls.append(recall)
        P = TP + FN
        plt.plot(conf_thresholds, recalls, lw=2, color=colors[area_i],
                     label=str(area_thresholds[area_i]) + '(' +  str(int(P)) + ')')  # 绘制每一条recall曲线
        plt.plot(conf_thresholds, recalls, 'o', color=colors[area_i])
    plt.legend(loc="lower left")
    # 画对角线
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlabel('Confidence')
    plt.ylabel('Recall')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Confidence-Recall')
    plt.grid()
    savename = data_mat[:data_mat.rfind("\\")] + "\\CRcurveObject.png"
    plt.savefig(savename)
    plt.show()


# colors = plt.cm.hsv(np.linspace(0, 1, 10)).tolist()
colors = ['Black', 'Blue', 'Cyan', 'Pink', 'Red', 'Purple', 'Gold', 'Chartreuse','Gray', 'Chocolate']
conf_thresholds = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95], dtype=np.float64)
ROOTDIR = "\\\\192.168.1.186/PedestrianData/"
all_change_group = []  # 初始化
s_ids = np.arange(len(conf_thresholds))
area_thresholds = np.array([0.0025, 0.005, 0.01, 0.015, 0.02, 0.04, 0.08, 0.1, 0.25, 1.0],dtype=np.float64) # area 区间
all_change_group =  [[] for x in range(len(area_thresholds))]  # 初始化
for k in range(0, len(area_thresholds), 1):
    for j in range(0, len(conf_thresholds), 1):
        all_change_group[k].append({'TP': 0, 'FN': 0})

if __name__ == "__main__":
    save_data("../Data_0922/val.txt", # 样本列表，注意这里的样本列表要与PR_statistic.py中样本列表相同！
              "COMPARE2/add_prior_gamma2_D1add15_new_P5N35D15E4_noSqrt/add_prior_gamma2_D1add15_new_P5N35D15E4_noSqrt_iter_200000") # PR_statistic.py中输出的目标检测结果

    # 绘制统计结果
    draw_curve("COMPARE2\\add_prior_gamma2_D1add15_new_P5N35D15E4_noSqrt\\add_prior_gamma2_D1add15_new_P5N35D15E4_noSqrt_iter_200000")