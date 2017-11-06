# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import xml.etree.cElementTree as et
import os
# setup plot details

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
@return: boxes, width, height
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
@param param2: 检测结果列表文件(无后缀）
@output: 检测结果同名的mat文件，记录FP,TP,FN
"""
def save_data(testList, resultList):
    PR_mat = resultList.strip() + '.mat'
    with open(testList) as fp1, open(resultList + '.txt') as fp2:  # 对于每个测试图片
        for testFile, resultFile in zip(fp1, fp2):
            xml_name = ROOTDIR + testFile.strip().split('.jpg ')[1] # xml文件完整路径
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

            for conf_i in range(0, len(conf_thresholds), 1):  # 对于每个分类置信阈值（从小到大排列的）
                TP = 0  # 正检
                FP = 0  # 误检

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
                            if (computIOU(boxT, result_box[1]) >= 0.5):  # 如果有任意一个检测框能和ground_truth_box 匹配上则TP+1
                                TP += 1  # 正确检测
                                break
                FN = len(true_boxes) - TP # 漏检

                all_change_group[conf_i]['TP'] += TP
                all_change_group[conf_i]['FP'] += FP
                all_change_group[conf_i]['FN'] += FN

    scipy.io.savemat(PR_mat,{ 'all_change_group': all_change_group})

"""
@function:绘制PR曲线
@param param1: 统计结果mat文件（不限制数量）
"""
def draw_curve(*curves):
    fig, axes = plt.subplots(nrows=1, figsize=(10, 8))
    for curve_i, curve_name in enumerate(curves):
        data = scipy.io.loadmat(curve_name)
        data = data['all_change_group'][0]
        Ps = []
        recalls = []
        precisions = []
        data_name = os.path.basename(curve_name)[:-12]# 图例名称
        for conf_i in range(0, len(conf_thresholds), 1):
            TP = float(data[conf_i]['TP'])
            FP = float(data[conf_i]['FP'])
            FN = float(data[conf_i]['FN'])
            if TP == 0:
                recall = 0
                precision = 0
            else:
                P = TP + FN
                recall = TP / (TP + FN)
                precision = TP / (TP + FP)
            Ps.append(P)
            recalls.append(recall)
            precisions.append(precision)
        axes.plot(recalls, precisions, lw=2, color=colors[curve_i],
                          label=data_name )  # 绘制每一条recall曲线
        plt.plot(recalls, precisions, 'o', color=colors[curve_i])
        for conf_i in range(0, len(conf_thresholds), 1):
            plt.annotate(conf_thresholds[conf_i], xy=(recalls[conf_i], precisions[conf_i]),
                         xytext=(recalls[conf_i], precisions[conf_i]),
                         # arrowprops=dict(facecolor="r", headlength=3, headwidth=3, width=1)
                         )
    plt.legend(loc="lower left")
    #画对角线
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.yticks(np.arange(0, 1.01, 0.1))  # 设置x轴刻度
    plt.xticks(np.arange(0, 1.01, 0.1))  # 设置x轴刻度
    plt.title('Precision-Recall')
    plt.grid()
    plt.show()


# colors = plt.cm.hsv(np.linspace(0, 1, 10)).tolist()
colors = ['Red', 'Blue', 'DeepSkyBlue', 'ForestGreen', 'HotPink',
          'Black', 'Purple', 'Gold', 'Brown', 'Violet', 'Cyan']
lw = 2
conf_thresholds = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95], dtype=np.float64)
ROOTDIR = "\\\\192.168.1.186/PedestrianData/"
all_change_group = []  # 初始化
for j in range(0, len(conf_thresholds), 1):
    all_change_group.append({'TP': 0, 'FP': 0, 'FN':0})
s_ids = np.arange(len(conf_thresholds))



if __name__ == "__main__":
    # save_data("..\\Data_0922\\val.txt", # 样本列表，注意这里的样本列表要与PR_statistic.py中样本列表相同！
    #           'COMPARE2\\add_prior_gamma2_D1_new_P5N35D15E4_noSqrt\\'
    #           'add_prior_gamma2_D1_new_P5N35D15E4_noSqrt_iter_200000') # PR_statistic.py中输出的目标检测结果

    # 曲线数量+各个曲线对应的统计结果文件
    draw_curve(
            "COMPARE2\\gamma2_D_new\\gamma2_D_new_iter_200000",
            "COMPARE2\\add_prior_gamma2_D_new_P5N4D15E4\\add_prior_gamma2_D_new_P5N4D15E4_iter_200000",
            "COMPARE2\\add_prior_gamma2_D_new_P5N4D15E4_noSqrt\\add_prior_gamma2_D_new_P5N4D15E4_noSqrt_iter_200000",
            "COMPARE2\\add_prior_gamma2_D1_new_P5N35D15E4_noSqrt\\add_prior_gamma2_D1_new_P5N35D15E4_noSqrt_iter_200000",
            # "COMPARE2\\add_prior_gamma2_D1add15_new_P5N35D15E4_noSqrt\\add_prior_gamma2_D1add15_new_P5N35D15E4_noSqrt_iter_190000"
            )