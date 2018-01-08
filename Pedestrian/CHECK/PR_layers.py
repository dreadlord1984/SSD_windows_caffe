# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.cElementTree as et
import prettyplotlib as ppl
import scipy.io
import os
import cv2

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


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
@function:判断prior box来自于那一层
@param param: prior box list
@return: layer list
"""
def which_layer(index_list):
    index_array = np.array(index_list)
    layer = np.zeros(index_array.size,  dtype=np.int32)
    min = 0
    max = layer_priorbox_num[0]
    for i in range(1, len(layer_priorbox_num), 1):
        min += layer_priorbox_num[i-1]
        max += layer_priorbox_num[i]
        mask =  (index_array >= min) & (index_array < max)
        layer[mask] = i
    return layer

"""
@function:统计检测结果按照gt的匹配来源划分到各个尺度检测器的TP/FP/FN
@param param1: 列表所在文件夹
@param param2: 经过网络+NMS后的检测结果（调用Test_ssd.bat实现）
@param param3: 匹配合并后的列表
@param param4: 可选参数，是否显示误检（默认不显示）
@return: layer list
"""
def save_data(resultDir, resultList, matchList, view = False):
    resultList = os.path.join(resultDir, resultList)
    matchList = os.path.join(resultDir, matchList)
    name, extension = os.path.splitext(resultList)
    savename = name + '_Layers.mat'
    image_num = 0
    with open(resultList) as fp1, open(matchList) as fp2:  # 对于每个测试图片
        for resultFile in fp1:  # 每一行匹配数据 resultFile
            image_num += 1
            priorFile = fp2.readline()  # 每一行检测数据 priorFile
            result_date = resultFile.strip().split('\t')
            prior_datas = priorFile.strip().split('\t')
            img_name = ROOTDIR + result_date[0]
            xml_name = ROOTDIR + result_date[0].replace('jpg','xml').replace('JPEGImages','Annotations')
            true_boxes, width, height = readXML(xml_name)  # 所有的ground truth boxes
            # 1.读取检测结果（NMS之后的）
            result_boxes_total = int(result_date[1])
            result_boxes = []
            for box_index in range(0, result_boxes_total, 1):
                prior_box_index = int(result_date[6 * box_index + 2])
                prior_box_score = float(result_date[6 * box_index + 3])
                prior_box_xmin = int(float(result_date[6 * box_index + 4]) * width)
                prior_box_ymin = int(float(result_date[6 * box_index + 5]) * height)
                prior_box_xmax = int(float(result_date[6 * box_index + 6]) * width)
                prior_box_ymax = int(float(result_date[6 * box_index + 7]) * height)
                result_boxes.append([prior_box_index, prior_box_score,
                                   [prior_box_xmin, prior_box_ymin,
                                   prior_box_xmax, prior_box_ymax]]
                                  )
            # 1.读取匹配结果
            match_box_list = [[] for x in range(len(true_boxes))]
            prior_boxes_total = int(prior_datas[1])  # 匹配box数量
            for i in range(0, prior_boxes_total, 1):
                gt_box_index = int(prior_datas[7 * i + 8])  # 当前匹配的gt box序号（从0开始）
                match_index = int(prior_datas[7 * i + 2])
                match_box_list[gt_box_index].append(match_index)

            #############################################################
            if view:
                image = plt.imread(img_name)
                plt.imshow(image)
                currentAxis = plt.gca()
            #############################################################

            # 2.统计所有的gt box的匹配来源
            layers_dic = []
            for indexT, boxT in enumerate(true_boxes):
                top_layer_T = which_layer(match_box_list[indexT])
                layer_dic = {}
                for item in top_layer_T:
                    if item in layer_dic.keys():
                        layer_dic[item] += 1
                    else:
                        layer_dic[item] = 1
                for key in layer_dic.keys():
                    all_gt_group[key] += layer_dic[key] / float(top_layer_T.size)
                layers_dic.append(layer_dic)

            # 对于每种分类置信度
            for conf_i in range(0, len(conf_thresholds), 1):  # 对于每个分类置信阈值
                # 3.选出检测器判断为正的
                select_boxes = [prior_box for prior_box in result_boxes if prior_box[1] >= conf_thresholds[conf_i]] #预测为正的所有prior box
                select_indices = [select_box[0] for select_box in select_boxes] # prior box 序号列表
                top_layer = which_layer(select_indices)
                # 4.正检和误检
                for index, result_box in enumerate(select_boxes):  # 对每个result box
                    not_match = 0
                    for boxT in true_boxes:
                        if (computIOU(boxT, result_box[2]) < 0.5):
                            not_match += 1  # 未匹配次数
                    if not_match == len(true_boxes):  # 没有一个gt box能和result box匹配则为误检FP
                        allFPs[top_layer[index]][conf_i]['FP'] += 1
                        if view:
                            display_txt = 'FP: %.2f' % (result_box[1])
                            coords = (result_box[2][0], result_box[2][1]), \
                                     result_box[2][2] - result_box[2][0] + 1, result_box[2][3] - result_box[2][1] + 1
                            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='red', linewidth=2))
                            currentAxis.text(result_box[2][0], result_box[2][1], display_txt, bbox={'facecolor': 'red', 'alpha': 0.5})
                    else:# 有至少一个gt box能和result box匹配则为正检TP
                        allFPs[top_layer[index]][conf_i]['TP'] += 1
                        if view:
                            display_txt = 'TP: %.2f' % (result_box[1])
                            coords = (result_box[2][0], result_box[2][1]), \
                                     result_box[2][2] - result_box[2][0] + 1, result_box[2][3] - result_box[2][1] + 1
                            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='green', linewidth=2))
                            currentAxis.text(result_box[2][0], result_box[2][1], display_txt, bbox={'facecolor': 'green', 'alpha': 0.5})
                # 5.漏检
                for index, boxT in enumerate(true_boxes):  # 对每个result box
                    TP = 0  # 正检
                    for result_box in select_boxes:
                        if (computIOU(boxT, result_box[2]) >= 0.5):  # 如果有任意一个检测框能和ground_truth_box 匹配上则TP+1
                            TP += 1  # 正确检测
                            break
                    if TP == 0:  # 漏检
                        total_match = sum(layers_dic[index].values())
                        for key, value in layers_dic[index].items():
                            allFPs[key][conf_i]['FN'] += value / float(total_match)
                        # display_txt = 'TP: %.2f' % (result_box[1])
                        # coords = (result_box[2][0], result_box[2][1]), \
                        #          result_box[2][2] - result_box[2][0] + 1, result_box[2][3] - result_box[2][1] + 1
                        # currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='yellow', linewidth=2))
                        # currentAxis.text(result_box[2][0], result_box[2][1], display_txt, bbox={'facecolor': 'yellow', 'alpha': 0.5})
                        # print 'hello'
            if view:
                plt.show()
            if image_num % 1000 == 0:
                print( '** ** ** ** ** ** process %d images ** ** ** ** ** ** ** ' % image_num)
    scipy.io.savemat(savename,{ 'allFPs': allFPs, 'allGts': all_gt_group})

def draw_curve(data_name, image_num):
    data_name = data_name.strip() + '_Layers.mat'
    data = scipy.io.loadmat(data_name)
    allFPs = data['allFPs']
    allGts = data['allGts'][0]
    # s_ids = np.arange(len(conf_thresholds))
    # labels = [conf_thresholds[i] for i in s_ids]
    for k in range(0, len(layer_priorbox_num), 1):
        recalls = []
        precisions = []
        for conf_i in range(0, len(conf_thresholds), 1):
            TP = float(allFPs[k][conf_i]['TP'])
            FP = float(allFPs[k][conf_i]['FP'])
            FN = float(allFPs[k][conf_i]['FN'])
            if TP == 0:
                recall = 0
                precision = 0
            else:
                recall = TP / (TP + FN)
                precision = TP / (TP + FP)
            recalls.append(recall)
            precisions.append(precision)
        plt.plot(recalls, precisions, lw=2, color=colors[k], label=[layer_priorbox_num[k],round(allGts[k],1)])  # 绘制每一条recall曲线
        plt.plot(recalls, precisions, 'o', color=colors[k])
        if k == 0:
            for conf_i in range(0, len(conf_thresholds), 1):
                plt.annotate(conf_thresholds[conf_i], xy=(recalls[conf_i], precisions[conf_i]),
                             xytext=(recalls[conf_i], precisions[conf_i]),color="#95a5a6",
                             # arrowprops=dict(facecolor='red', headlength=3, headwidth=3, width=1)
                             )
    plt.title('SOFTMAX Precision-Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle = "--", color=(0.6, 0.6, 0.6))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    savename1 = data_name.replace('mat','jpg')
    plt.savefig(savename1)
    plt.show()

##################################################################################
ROOTDIR = "\\\\192.168.1.186\\PedestrianData\\" # 待测试样本集所在根目录
resize_width = 384
resize_height = 256
## 注意！这里layer_priorbox_num，不同的网络可能不同！
# layer_priorbox_num = np.array([15360, 1920, 480, 120, 30, 5],dtype=np.int32) # layer层priorbox 数
# layer_priorbox_num = np.array([9216, 2304, 576, 144, 36, 6],dtype=np.int32) # layer层priorbox 数
layer_priorbox_num = np.array([7680, 1920, 480, 120, 30, 5],dtype=np.int32) # layer层priorbox 数
##################################################################################
conf_thresholds = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95], dtype=np.float64)
all_gt_group = np.zeros(layer_priorbox_num.size,dtype=np.float)
allFPs =  [[] for x in range(len(layer_priorbox_num))]  # 误检初始化
for k in range(0, len(layer_priorbox_num), 1):
    for j in range(0, len(conf_thresholds), 1):
        allFPs[k].append({'TP': 0, 'FP':0,  'FN':0}) # 正检、误检
colors = ['Black', 'Blue', 'Cyan', 'Pink', 'Red', 'Purple', 'Gold', 'Chartreuse']


if __name__ == "__main__":
    save_data(
        "..\\Data_0922\\FocalLoss_NONE_D1_noSqrt", # 文件夹
        "RESULT_VAL_image_list.txt", # 验证集检测结果列表
        "IOU_ALL_VAL_image_List.txt", # 验证集匹配结果列表
        view = False
              )

    # 曲线数量+各个曲线对应的统计结果文件
    draw_curve("..\\Data_0922\\FocalLoss_NONE_D1_noSqrt\\RESULT_VAL_image_list", 8765)
    # draw_curve("..\\Data_0922\\FocalLoss_NONE_D\\RESULT_VAL_image_list", 8765)