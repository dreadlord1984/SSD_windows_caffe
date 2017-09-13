# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import xml.etree.cElementTree as et
import scipy.io

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
@function:将匹配合并结果和检测结果比较回归IOU变化
@param param1: 匹配合并列表文件
@param param2: 检测结果列表文件
@param param3: 待保存IOU变化文件
"""
def save_data(priorList, resultList, data_mat):
    with open(priorList) as fp1, open(resultList) as fp2: # 对于每个测试图片
        plt.close('all')
        for resultFile in fp2: # 每一行匹配数据 resultFile
            priorFile = fp1.readline() # 每一行检测数据 priorFile
            prior_datas = priorFile.strip().split('\t')
            result_datas = resultFile.strip().split('\t')
            # img_name = ROOTDIR + prior_datas[0]
            xml_name = ROOTDIR + prior_datas[1]
            # image = plt.imread(img_name)
            # width = image.shape[1]
            # height = image.shape[0]
            true_boxes, width, height = readXML(xml_name) # 所有的ground truth boxes
            prior_IOU = []  # 训练时所有匹配的prior box与gt box的IOU
            prior_boxes = [] # 训练时所有匹配的prior box坐标 [[prior_box_index, [xmin, ymin, xmax, ymax]]...]
            gt_boxes = [] # 样本所有gt box坐标 [[gt_box_index, [xmin, ymin, xmax, ymax]]...]
            prior_boxes_total = int(prior_datas[2]) # 匹配box数量
            for i in range(0, prior_boxes_total, 1):
                prior_box_index = int(prior_datas[7 * i + 3]) # 当前匹配proir box序号（从0开始）
                gt_box_index = int(prior_datas[7 * i + 9]) # 当前匹配的gt box序号（从0开始）
                prior_box_coordinates = [float(prior_datas[7*i + 5]) * width, float(prior_datas[7*i + 6]) * height, float(prior_datas[7*i + 7]) * width, float(prior_datas[7*i + 8]) * height]
                prior_boxes.append([prior_box_index, prior_box_coordinates])
                gt_boxes.append([gt_box_index, true_boxes[gt_box_index]])
                prior_IOU.append(float(prior_datas[7*i + 4]))
            if(len(prior_IOU)!= len(prior_boxes) or len(prior_IOU)!= len(gt_boxes)):
                print "匹配维度不统一！"
                break

            result_boxes = [] # 所有prior box的回归结果
            result_IOU = []  # result box与gt box的IOU
            reslut_conf = []  # 检测得到result box置信度
            result_boxes_total = (len(result_datas)-1)/6  # 检测得到的box数量
            for j in range(0, result_boxes_total, 1):
                result_box_index = int(result_datas[6 * j + 1])  # 当前匹配proir box序号（从0开始）
                result_box_coordinates = [float(result_datas[6 * j + 3]) * width, float(result_datas[6 * j + 4]) * height,
                                         float(result_datas[6 * j + 5]) * width, float(result_datas[6 * j + 6]) * height]
                result_boxes.append([result_box_index, result_box_coordinates])
                result_IOU.append(computIOU(gt_boxes[j][1], result_box_coordinates))
                reslut_conf.append(float(result_datas[6 * j + 2]))

            change_IOU = np.subtract(result_IOU, prior_IOU) # IOU变化
            for index in range(0, len(change_IOU), 1):
                div = int(prior_IOU[index] / min_threshold) # 落在哪个IOU区间
                scope = len(change_scope)/2 + int(change_IOU[index] / min_scope) # IOU变化区间
                all_change_group[div][scope] += 1

    scipy.io.savemat(data_mat, mdict={'IOU_change_statistic': all_change_group})

"""
@function:为直方图添加y轴信息
"""
def autolabel(rects, Num=1.12, rotation1=90, NN=1):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() - 0.01 + rect.get_width() / 2., Num * height, '%s' % int(height * NN),
                 rotation=rotation1)

"""
@function:将IOU变化以图形形式展现
@param param1: IOU变化文件
"""
def show_hist(data_mat):
    all_change_group = scipy.io.loadmat(data_mat)
    all_change_group = all_change_group['IOU_change_statistic']
    color = ['k', 'b', 'g', 'r', 'm']
    for i in range(0, len(all_change_group), 1):
        plt.subplot(151+i) # 放在一个图片里显示
        # plt.subplots(1)
        plt.plot(change_scope, all_change_group[i], lw=2,color=color[i])
        rects = plt.bar(left=change_scope, height=all_change_group[i], color='lightyellow',
                        width=0.05, align="center", yerr=0.000001)
        #autolabel(rects, 1.0) # 为每个直方图加标注
        plt.xlim((-1, 1))
        plt.title('IOU_change: '+ '%.1f' % thresholds[i])
        plt.ylabel('Num')
        plt.xlabel('IOU change')
        plt.grid()
    plt.show()

#matplotlib.rcParams['figure.figsize'] = (6, 8)  # 设定显示大小
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
min_threshold = 0.2
thresholds = np.linspace( min_threshold, 1, 5 ) # IOU 区间
min_scope = 0.05
change_scope = np.arange( -0.95, 1.01, min_scope ) # IOU变化区间
all_change_group = [[0 for x in range(len(change_scope))] for y in range(len(thresholds))] # 初始化为0

ROOTDIR = "\\\\192.168.1.186/PedestrianData/" # 样本所在根目录

if __name__ == "__main__":
    save_data("../Data_0825/IOU_ALL_image_List.txt",
              "../View/COMPARE/MAX_NEGATIVE_A75G20_S/result_ALL_image_List.txt",
              "../View/COMPARE/MAX_NEGATIVE_A75G20_S/IOU_change_curve.mat")
    show_hist("../View/COMPARE/MAX_NEGATIVE_A75G20_S/IOU_change_curve.mat")