# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.cElementTree as et
import prettyplotlib as ppl
import scipy.io

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2


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


def save_data(imgList, model_def, model_weights, savename):
    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104,117,123])) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    # set net to batch size of 1
    resize_width = 384
    resize_height = 256
    net.blobs['data'].reshape(1,3,resize_height,resize_width)

    image_num = 0
    for imgFile in open(imgList).readlines():  # 对于每个测试图片
        image_num += 1
        img_name = ROOTDIR + imgFile.strip().split('.jpg ')[0]
        xml_name = ROOTDIR + imgFile.strip().split('.jpg ')[1]
        image = caffe.io.load_image(img_name+'.jpg')
        true_boxes, width, height = readXML(xml_name)  # 所有的ground truth boxes
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image
        # Forward pass.
        detections = net.forward()['detection_out']

        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        plt.imshow(image)
        currentAxis = plt.gca()

        for conf_i in range(0, len(conf_thresholds), 1):  # 对于每个分类置信阈值
            conf_i = 4
            # 1.Get detections with confidence higher than conf_threshold.
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresholds[conf_i]] #预测为正的所有default box

            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_labels = get_labelname(labelmap, top_label_indices)
            top_xmin = det_xmin[top_indices]
            top_ymin = det_ymin[top_indices]
            top_xmax = det_xmax[top_indices]
            top_ymax = det_ymax[top_indices]
            top_layer = which_layer(top_indices)

            result_boxes = []
            for i in xrange(top_conf.shape[0]): # 对每个检测到的目标
                xmin = max(int(round(top_xmin[i] * image.shape[1])),0)
                ymin = max(int(round(top_ymin[i] * image.shape[0])),0)
                xmax = min(int(round(top_xmax[i] * image.shape[1])), image.shape[1])
                ymax = min(int(round(top_ymax[i] * image.shape[0])), image.shape[0])
                score = top_conf[i]
                layer_index = top_layer[i]
                result_boxes.append([layer_index, score, [xmin, ymin, xmax, ymax]])

            for result_box in result_boxes:  # 对每个result box
                not_match = 0
                for boxT in true_boxes:
                    if (computIOU(boxT, result_box[2]) < 0.5):
                        not_match += 1  # 未匹配次数
                if not_match == len(true_boxes):  # 没有一个gt box能和result box匹配则为误检FP
                    allFPs[result_box[0]][conf_i]['FP'] += 1
                    # display_txt = 'TP: %.2f' % (result_box[1])
                    # coords = (result_box[2][0], result_box[2][1]), \
                    #          result_box[2][2] - result_box[2][0] + 1, result_box[2][3] - result_box[2][1] + 1
                    # currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='green', linewidth=2))
                    # currentAxis.text(result_box[2][0], result_box[2][1], display_txt,
                    #                  bbox={'facecolor': 'blue', 'alpha': 0.5})
                else:# 有至少一个gt box能和result box匹配则为正检TP
                    allFPs[result_box[0]][conf_i]['TP'] += 1
                    # display_txt = 'TP: %.2f' % (result_box[1])
                    # coords = (result_box[2][0], result_box[2][1]), \
                    #          result_box[2][2] - result_box[2][0] + 1, result_box[2][3] - result_box[2][1] + 1
                    # currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='green', linewidth=2))
                    # currentAxis.text(result_box[2][0], result_box[2][1], display_txt, bbox={'facecolor': 'green', 'alpha': 0.5})

            # 2.Get detections with confidence lower than conf_threshold.
            top_indices = [i for i, conf in enumerate(det_conf) if conf < conf_thresholds[conf_i]]  # 预测为负的所有default box

            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_labels = get_labelname(labelmap, top_label_indices)
            top_xmin = det_xmin[top_indices]
            top_ymin = det_ymin[top_indices]
            top_xmax = det_xmax[top_indices]
            top_ymax = det_ymax[top_indices]
            top_layer = which_layer(top_indices)

            result_boxes = []
            for i in xrange(top_conf.shape[0]):  # 对每个检测到的目标
                xmin = max(int(round(top_xmin[i] * image.shape[1])), 0)
                ymin = max(int(round(top_ymin[i] * image.shape[0])), 0)
                xmax = min(int(round(top_xmax[i] * image.shape[1])), image.shape[1])
                ymax = min(int(round(top_ymax[i] * image.shape[0])), image.shape[0])
                score = top_conf[i]
                layer_index = top_layer[i]
                result_boxes.append([layer_index, score, [xmin, ymin, xmax, ymax]])

            for result_box in result_boxes:  # 对每个result box
                not_match = 0
                for boxT in true_boxes:
                    if (computIOU(boxT, result_box[2]) < 0.5):
                        not_match += 1  # 未匹配次数
                if not_match == len(true_boxes):  # 没有一个gt box能和result box匹配则为负检TN
                    allFPs[result_box[0]][conf_i]['TN'] += 1
                    # display_txt = 'TP: %.2f' % (result_box[1])
                    # coords = (result_box[2][0], result_box[2][1]), \
                    #          result_box[2][2] - result_box[2][0] + 1, result_box[2][3] - result_box[2][1] + 1
                    # currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='green', linewidth=2))
                    # currentAxis.text(result_box[2][0], result_box[2][1], display_txt,
                    #                  bbox={'facecolor': 'blue', 'alpha': 0.5})
                else:  # 有至少一个gt box能和result box匹配则为漏检FN
                    allFPs[result_box[0]][conf_i]['FN'] += 1
                    display_txt = 'FN: %.2f' % (result_box[1])
                    coords = (result_box[2][0], result_box[2][1]), \
                             result_box[2][2] - result_box[2][0] + 1, result_box[2][3] - result_box[2][1] + 1
                    currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='red', linewidth=2))
                    currentAxis.text(result_box[2][0], result_box[2][1], display_txt, bbox={'facecolor': 'red', 'alpha': 0.5})
            plt.show()
        if image_num % 1000 == 0:
            print( '** ** ** ** ** ** process %d images ** ** ** ** ** ** ** ' % image_num)
    scipy.io.savemat(savename,{ 'allFPs': allFPs})

def draw_curve(data_name, image_num):
    data = scipy.io.loadmat(data_name)
    allFPs = data['allFPs']
    s_ids = np.arange(len(conf_thresholds))
    labels = [conf_thresholds[i] for i in s_ids]
    FPr = [[] for x in range(len(layer_priorbox_num))]
    fig, axes = plt.subplots(nrows=2, ncols= 3, figsize=(36, 12))
    for k in range(0, len(layer_priorbox_num), 1):
        for conf_i in range(0, len(conf_thresholds), 1):
            FPr[k].append(float(allFPs[k][conf_i]['FP'])*100/(float(allFPs[k][conf_i]['FP'])+float(allFPs[k][conf_i]['TN'])))
        ppl.bar(axes[k/3][k%3],s_ids, FPr[k],
                    annotate=True,width = 0.4,
                    grid='y', xticklabels=labels,
                    color=colors[k])  # 绘制每一条recall曲线
        axes[k / 3][k % 3].set_title(layer_priorbox_num[k])
        axes[k / 3][k % 3].set_ylabel('FPro, %')
        axes[k / 3][k % 3].set_xlabel('confidence')
    plt.grid()
    savename1 = data_name.split('.mat')[0] + '_FPr_conf.png'
    plt.savefig(savename1)

    fig, axes = plt.subplots(nrows=2, ncols= 3, figsize=(36, 12))
    FNr = [[] for x in range(len(layer_priorbox_num))]
    for k in range(0, len(layer_priorbox_num), 1):
        for conf_i in range(0, len(conf_thresholds), 1):
            if allFPs[k][conf_i]['TN'] == 0:
                FNr[k].append(0)
            else:
                FNr[k].append(float(allFPs[k][conf_i]['FN']) * 100
                                   /float(allFPs[k][conf_i]['TP']  + allFPs[k][conf_i]['FN']))
        ppl.bar(axes[k/3][k%3],s_ids, FNr[k],
                    annotate=True,width = 0.4,
                    grid='y', xticklabels=labels,
                    color=colors[k])  # 绘制每一条recall曲线
        axes[k / 3][k % 3].set_title(layer_priorbox_num[k])
        axes[k / 3][k % 3].set_ylabel('FNs, %')
        axes[k / 3][k % 3].set_xlabel('confidence')
    plt.grid()
    savename2 = data_name.split('.mat')[0] + '_FNr_conf.png'
    plt.savefig(savename2)
    plt.show()

# load PASCAL VOC labels
labelmap_file = 'labelmap_VehicleFull.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)
# load model
ROOTDIR = "\\\\192.168.1.186/PedestrianData/" # 待测试样本集所在根目录

# configure
colors = ['Black', 'Blue', 'Cyan', 'Pink', 'Red', 'Purple', 'Gold', 'Chartreuse']
layer_priorbox_num = np.array([9216, 2304, 576, 144, 36, 6],dtype=np.int32) # layer层priorbox 数
conf_thresholds = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95], dtype=np.float64)
allFPs =  [[] for x in range(len(layer_priorbox_num))]  # 误检初始化
for k in range(0, len(layer_priorbox_num), 1):
    for j in range(0, len(conf_thresholds), 1):
        allFPs[k].append({'TP': 0, 'TN':0, 'FP':0, 'FN':0}) # 正检、负检、误检、漏检

if __name__ == "__main__":
    save_data("../Data_0825/val.txt", # 样本列表
              '../CHECK/deploy2.prototxt',  # 检测网络,使用CHECK文件夹下的，不使用NMS和keep_top_k
              'COMPARE\NONE_A75G20_S_D\NONE_A75G20_S_D_iter_200000.caffemodel', # 模型
              'COMPARE\NONE_A75G20_S_D_fix\layers.mat') # 待输出的统计结果，即不同conf阈值下的TP、FP

    # 曲线数量+各个曲线对应的统计结果文件
    # draw_curve("COMPARE\NONE_A75G20_S_D\layers.mat", 6300)

    #  'COMPARE\NONE_A75G20_S_D_fix\NONE_A75G20_S_D_fix_iter_150000.caffemodel', # 模型