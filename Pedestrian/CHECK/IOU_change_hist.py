# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.cElementTree as et
import matplotlib
import matplotlib.mlab as mlab
import scipy.io

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
thresholds = np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float64)
# change_scope = np.array([-0.95, -0.9, -0.85, -0.8, -0.75, -0.7, -0.65, -0.6, -0.55, -0.5, -0.45, -0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0,
#                          0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0], dtype=np.float64)
change_scope = np.arange( -0.95, 1.01, 0.05 )
all_change_group = [[]  for y in range(len(thresholds))]

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

def readXML(xml_name):
    tree = et.parse(xml_name) #打开xml文档
    # 得到文档元素对象
    root = tree.getroot()
    size = root.find('size')  # 找到root节点下的size节点
    width = size.find('width').text  # 子节点下节点width的值
    height = size.find('height').text  # 子节点下节点height的值

    boundingBox = []
    for object in root.findall('object'):  # 找到root节点下的所有object节点
        bndbox = object.find('bndbox')  # 子节点下属性bndbox的值
        xmin = bndbox.find('xmin').text
        ymin = bndbox.find('ymin').text
        xmax = bndbox.find('xmax').text
        ymax = bndbox.find('ymax').text
        boundingBox.append([int(xmin), int(ymin), int(xmax), int(ymax)])
    return boundingBox

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



ROOTDIR = "\\\\192.168.1.186/PedestrianData/"

def save_data(priorList, resultList, hist_mat):
    with open(priorList) as fp1, open(resultList) as fp2: # 对于每个测试图片
        plt.close('all')
        for resultFile in fp2: # 每一行匹配数据 resultFile
            priorFile = fp1.readline() # 每一行检测数据 priorFile
            prior_datas = priorFile.strip().split('\t')
            result_datas = resultFile.strip().split('\t')
            img_name = ROOTDIR + prior_datas[0]
            xml_name = ROOTDIR + prior_datas[1]
            image = plt.imread(img_name)
            width = image.shape[1]
            height = image.shape[0]
            true_boxes = readXML(xml_name) # 所有的ground truth boxes
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

            change_IOU = np.subtract(result_IOU, prior_IOU)
            index = 0
            for index in range(0, len(prior_boxes), 1):
                div = int(prior_IOU[index] / thresholds[0])
                all_change_group[div].append(change_IOU[index])

    scipy.io.savemat(hist_mat, {'IOU_change': all_change_group})

def show_hist(hist_mat):
    all_change_group =  scipy.io.loadmat(hist_mat)
    all_change_group = all_change_group['IOU_change'][0]
    color = ['k', 'b', 'g', 'r', 'y']
    matplotlib.rcParams['figure.figsize'] = (5, 8)  # 设定显示大小
    for i in range(0, len(all_change_group), 1):
        fig, ax = plt.subplots(1)
        if len(all_change_group[i])==0:
            n, bins, patches = plt.hist(all_change_group[i],bins=40, range = (-1,1), normed=0, facecolor='g', alpha=0.75)
        else:
            n, bins, patches = plt.hist(all_change_group[i][0], bins=40, range=(-1, 1), normed=0, facecolor='g',
                                        alpha=0.75)
        # ppl.plot(change_scope, all_change_group[i], lw=2,color=color[i])
        #plt.ylim((0, 300))
        plt.xlim((-1, 1))
        ax.set_title('IOU_change: '+ '%.1f' % thresholds[i])
        ax.set_ylabel('Num')
        ax.set_xlabel('IOU change')
        plt.grid()
    plt.show()

if __name__ == "__main__":
    save_data("Data_0810/IOU_ALL_image_List.txt", "Data_0810/result_ALL_image_List.txt", "Data_0810/IOU_change.mat")
    show_hist("Data_0810/IOU_change.mat")