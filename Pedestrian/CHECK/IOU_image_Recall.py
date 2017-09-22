# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.cElementTree as et
import matplotlib
import prettyplotlib as ppl

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
thresholds = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0], dtype=np.float64)


TPs = np.zeros((len(thresholds)), dtype=np.float64) # 正检
FNs = np.zeros((len(thresholds)), dtype=np.float64) # 漏检
s_ids = np.arange(thresholds.size)

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

model_def = 'deploy.prototxt'
model_weights = 'snapshot_iter_120000.caffemodel'

ROOTDIR = "\\\\192.168.1.186/PedestrianData/"
priorList = "../Data_0810/IOU_ALL_image_List.txt"
resultList = "../Data_0810/result_ALL_image_List.txt"

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
        all_IOU = []  # 训练时所有匹配的prior box与gt box的IOU
        prior_boxes = [] # 训练时所有匹配的prior box坐标 [[prior_box_index, [xmin, ymin, xmax, ymax]]...]
        gt_boxes = [] # 样本所有gt box坐标 [[gt_box_index, [xmin, ymin, xmax, ymax]]...]
        reslut_conf = []  # 检测得到box置信度
        result_boxes = []
        prior_boxes_total = int(prior_datas[2]) # 匹配box数量
        for i in range(0, prior_boxes_total, 1):
            prior_box_index = int(prior_datas[7 * i + 3]) # 当前匹配proir box序号（从0开始）
            gt_box_index = int(prior_datas[7 * i + 9]) # 当前匹配的gt box序号（从0开始）
            prior_box_coordinates = [float(prior_datas[7*i + 5]) * width, float(prior_datas[7*i + 6]) * height, float(prior_datas[7*i + 7]) * width, float(prior_datas[7*i + 8]) * height]
            prior_boxes.append([prior_box_index, prior_box_coordinates])
            gt_boxes.append([gt_box_index, true_boxes[gt_box_index]])
            all_IOU.append(float(prior_datas[7*i + 4]))
        if(len(all_IOU)!= len(prior_boxes) or len(all_IOU)!= len(gt_boxes)):
            print "匹配维度不统一！"
            break

        result_boxes_total = (len(result_datas)-1)/6  # 检测得到的box数量
        for j in range(0, result_boxes_total, 1):
            result_box_index = int(result_datas[6 * j + 1])  # 当前匹配proir box序号（从0开始）
            result_box_coordinates = [float(result_datas[6 * j + 3]) * width, float(result_datas[6 * j + 4]) * height,
                                     float(result_datas[6 * j + 5]) * width, float(result_datas[6 * j + 6]) * height]
            result_boxes.append([result_box_index, result_box_coordinates])
            reslut_conf.append(float(result_datas[6 * j + 2]))

        # prior_boxes与result_boxes匹配
        prior_matching = np.zeros((len(prior_boxes)), dtype=np.int) - 1
        prior_index = 0
        for boxP in prior_boxes:
            result_index = 0
            for boxR in result_boxes:
                if(int(boxR[0]) == int(boxP[0])):
                    prior_matching[prior_index] = result_index
                result_index += 1
            prior_index += 1

        # 匹配结果与gt_boxes匹配
        index = 0  # prior_boxes中第几个box
        for boxP in prior_boxes:
            div = int(all_IOU[index] / thresholds[0])
            if prior_matching[index] < 0:
                FNs[div] += 1
            else:
                if (computIOU(gt_boxes[index][1], result_boxes[prior_matching[index]][1]) > 0.5):
                    TPs[div] += 1 # 正确检测
                else:
                    FNs[div] += 1
            index += 1
    print img_name.decode("gb2312")

print 'TPs: ', TPs
print 'FNs: ', FNs

matplotlib.rcParams['figure.figsize'] = (15, 5)  # 设定显示大小
fig, ax = plt.subplots(1)
labels = [thresholds[i] for i in s_ids]
recall2 = np.divide(TPs, np.add(TPs, FNs))
anno_area2s = [('%f' % a) for a in recall2[s_ids]]
ppl.bar(ax, np.arange(len(recall2)), recall2[s_ids], annotate=anno_area2s, grid='y', xticklabels=labels)
plt.xticks(rotation=25)
ax.set_title('(ALL_IOU_recall)')
ax.set_ylabel('Recall')
plt.savefig('../Data_0810/ALL_IOU_recall.png')
plt.show()