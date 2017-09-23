# -*- coding: utf-8 -*-
import numpy as np
import pylab as plt
import prettyplotlib as ppl
import xml.etree.cElementTree as et
import scipy.io

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
@function:将匹配合并结果和检测结果比较分类confidence变化
@param param1: 匹配合并列表文件
@param param2: 检测结果列表文件
@param param3: 待保存mat矩阵
"""
def save_data(priorList, resultList, data_mat):
    with open(priorList) as fp1, open(resultList) as fp2:  # 对于每个测试图片
        for priorFile in fp1:  # 每一行匹配数据 resultFile
            resultFile = fp2.readline()  # 每一行检测数据 priorFile
            prior_datas = priorFile.strip().split('\t')
            result_datas = resultFile.strip().split('\t')
            xml_name = ROOTDIR + prior_datas[1]
            # print prior_datas[1]
            # 1.记录各个gt box面积占比
            true_boxes, width, height = readXML(xml_name)  # 所有的ground truth boxes
            gt_areas = []
            gt_box_index = 0
            for gt_box in true_boxes:
                area_ratio = float((gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])) / (width * height)
                gt_areas.append([gt_box_index, area_ratio])
                gt_box_index += 1

            # 2.记录各个prior box匹配IOU、分类conf和回归IOU
            prior_IOU = []  # 训练时所有匹配的prior box与gt box的IOU
            prior_conf = []  # 训练时所有匹配的prior box与gt box的IOU
            prior_boxes_total = int(prior_datas[2])  # 匹配box数量
            result_IOU = []  # result box与gt box的IOU
            # total_prior_num += prior_boxes_total
            for i in range(0, prior_boxes_total, 1):
                gt_box_index = int(prior_datas[7 * i + 9])  # 当前匹配的gt box序号（从0开始）
                IOU = float(prior_datas[7 * i + 4])
                prior_IOU.append([gt_box_index, IOU])
                conf = float(result_datas[6 * i + 2])  # 分类置信度
                prior_conf.append([gt_box_index, conf])
                result_box_coordinates = [float(result_datas[6 * i + 3]) * width, float(result_datas[6 * i + 4]) * height,
                                          float(result_datas[6 * i + 5]) * width, float(result_datas[6 * i + 6]) * height]
                result_IOU.append([gt_box_index, computIOU(true_boxes[gt_box_index], result_box_coordinates)])

            # 3.按照area和匹配IOU统计划分
            for n in range(0, prior_boxes_total, 1):  # 对于每个prior box
                    for area_i in range(0, len(area_thresholds), 1):  # 判断area区间段
                        if (gt_areas[prior_IOU[n][0]][1] <= area_thresholds[area_i]):
                            area_index = area_i
                            break
                    for IOU_i in range(0, len(IOU_thresholds), 1):  # 判断IOU区间段
                        if (prior_IOU[n][1] <= IOU_thresholds[IOU_i]):
                            IOU_index = IOU_i
                            break
                    for IOU_i in range(0, len(IOU_thresholds), 1):  # 判断IOU区间段
                        if (result_IOU[n][1] <= IOU_thresholds[IOU_i]):
                            rIOU_index = IOU_i
                            break
                    area_prior_num[area_index][IOU_index] += 1
                    area_result_num[area_index][rIOU_index] += 1
                    for conf_i in range(0, len(conf_thresholds), 1):  # 对于每个分类置信阈值
                        if prior_conf[n][1] >= conf_thresholds[conf_i]:
                            all_change_group[area_index][conf_i][IOU_index]['Pos'] += 1
                        else:
                            all_change_group[area_index][conf_i][IOU_index]['Neg'] += 1
    # 3.将数据保存
    scipy.io.savemat(data_mat, {'prior_rgression_statistic': prior_rgression_statistic, 'all_change_group': all_change_group})

"""
@function:
1.将gt boxes和prior boxes分布绘制
2.将各个conf阈值条件下confidence的变化以recall曲线展现
@param param1: gt boxes和prior boxes分布
@param param2: confidence变化文件
"""
def draw_curve(data_mat):
    data = scipy.io.loadmat(data_mat)
    prior_rgression_statistic = data['prior_rgression_statistic']
    area_prior_num = prior_rgression_statistic[0]
    area_result_num = prior_rgression_statistic[1]
    all_change_group = data['all_change_group']
    # 1. 绘制gt boxes和prior boxes分布
    for pl in range(0, len(area_thresholds), 1):
        total_prior_num = 0
        for num in area_prior_num[pl]:
            total_prior_num += num
        fig, axes = plt.subplots(nrows=2, figsize=(8, 12))
        strand_names = 'prior box distribution:%d' % total_prior_num
        yalbel_names = 'area: ' + '%.4f' % area_thresholds[pl] + ' prioe box num'
        labels = [IOU_thresholds[i] for i in s_ids]
        ppl.bar(axes[0], s_ids, area_prior_num[pl],
                annotate=True,width = 0.4,
                grid='y', xticklabels=labels,
                color=colors[4]) #color=[colors[i] for i in s_ids]
        ppl.bar(axes[0], s_ids+0.4, area_result_num[pl],
                annotate=True,width = 0.4,
                grid='y',
                color=colors[0])
        axes[0].set_title(strand_names)
        axes[0].set_ylabel(yalbel_names)
        axes[0].set_xlabel('IOU')

        # 2. 绘制不同confidence阈值下的recall曲线
        prior_num = []
        for conf_i in range(0, len(conf_thresholds), 1):
            recalls = []
            for j in range(0, len(IOU_thresholds), 1):
                if conf_i == 0:
                    prior_num.append(all_change_group[pl][conf_i][j]['Neg'] + all_change_group[pl][conf_i][j]['Pos'])
                TP = float(all_change_group[pl][conf_i][j]['Pos'])
                FP = float(all_change_group[pl][conf_i][j]['Neg'])
                if TP == 0:
                    recall = 0
                else:
                    recall = TP / (TP + FP)
                recalls.append(recall)
            axes[1].plot(s_ids, recalls, lw=2, color=colors[conf_i],
                         label=str(conf_thresholds[conf_i]))  # 绘制每一条recall曲线
            plt.annotate(conf_thresholds[conf_i], xy=(s_ids[len(s_ids) / 2], recalls[len(recalls) / 2]),
                         xytext=(s_ids[len(s_ids) / 2], recalls[len(recalls) / 2]),
                         arrowprops=dict(facecolor="r", headlength=5, headwidth=5, width=2))
        plt.grid()
        plt.xticks(s_ids, IOU_thresholds)
        plt.xlabel('IOU')
        plt.ylabel('area: ' + '%.4f' % area_thresholds[pl] + ' Recall')
        plt.title('IOU-Recall')
        plt.legend(loc="upper left")
        plt.ylim((0, 1))
        # ax2 = axes[1].twiny()
        # plt.xticks(s_ids, prior_num, rotation=10)
        savename = data_mat[:data_mat.rfind("\\")] + '\\figure_' + str(pl) + '.png';
        plt.savefig(savename)
        plt.show()

ROOTDIR = "\\\\192.168.1.186/PedestrianData/" # 样本所在根目录
min_conf_threshold = 0.1 # 最小分类置信度阈值
conf_thresholds = np.linspace( min_conf_threshold, 1, 10 ) # 分类置信度阈值
min_threshold = 0.1 # 最小IOU区间
IOU_thresholds = np.linspace( min_threshold, 1, 10 ) # IOU 区间
area_thresholds = np.array([0.0025, 0.005, 0.01, 0.015, 0.02, 0.04, 0.08, 0.1, 0.25, 1.0],dtype=np.float64) # area 区间
area_prior_num = np.zeros((len(area_thresholds), len(IOU_thresholds)),dtype=np.int32)
area_result_num = np.zeros((len(area_thresholds), len(IOU_thresholds)),dtype=np.int32)
prior_rgression_statistic = []
prior_rgression_statistic.append(area_prior_num)
prior_rgression_statistic.append(area_result_num)
all_change_group =  [[[] for x in range(len(conf_thresholds))] for y in range(len(area_thresholds))]  # 初始化
for k in range(0, len(area_thresholds), 1):
    for j in range(0, len(conf_thresholds), 1):
        for i in range(0, len(IOU_thresholds), 1):
            all_change_group[k][j].append({'Neg': 0, 'Pos': 0})
s_ids = np.arange(len(IOU_thresholds))
colors = plt.cm.hsv(np.linspace(0, 1, 10)).tolist()

if __name__ == "__main__":
    save_data("..\\Data_0825\\IOU_ALL_image_List2.txt",
              "..\\View\\COMPARE\\0919\\result_ALL_image_List.txt",
              "..\\View\\COMPARE\\0919\\object_confidence_IOU_change_curve.mat")
    draw_curve("..\\View\\COMPARE\\0919\\object_confidence_IOU_change_curve.mat")
