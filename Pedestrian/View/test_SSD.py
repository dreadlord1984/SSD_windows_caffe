# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.cElementTree as et

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load PASCAL VOC labels
labelmap_file = 'labelmap_VehicleFull.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

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
model_weights = 'snapshot_iter_110000.caffemodel'
ROOTDIR = "\\\\192.168.1.186/PedestrianData/"
imgList = "../Data_0731/val.txt"

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

TPs = 0 # 正检
FPs = 0 # 误检
FNs = 0 # 漏检

for imgFile in open(imgList).readlines():  # 对于每个测试图片
    img_name = ROOTDIR + imgFile.strip().split('.jpg ')[0]
    xml_name = ROOTDIR + imgFile.strip().split('.jpg ')[1]
    image = caffe.io.load_image(img_name+'.jpg')
    true_boxes = readXML(xml_name);
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

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.2]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

    plt.imshow(image)
    currentAxis = plt.gca()

    TP = 0 # 正检
    FP = 0 # 误检
    FN = 0 # 漏检
    detectBoxes = []
    for i in xrange(top_conf.shape[0]): # 对每个检测到的目标
        not_match = 0
        xmin = int(round(top_xmin[i] * image.shape[1]))
        ymin = int(round(top_ymin[i] * image.shape[0]))
        xmax = int(round(top_xmax[i] * image.shape[1]))
        ymax = int(round(top_ymax[i] * image.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = top_labels[i]
        display_txt = '%s: %.2f'%(label_name, score)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[label]
        detectBoxes.append([xmin, ymin, xmax, ymax])
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})
        for boxT in true_boxes:
            if (computIOU(boxT, [xmin, ymin, xmax, ymax]) < 0.5):
                not_match += 1  # 未匹配次数
        if not_match == len(true_boxes):
            FP += 1

    for boxT in true_boxes:
        currentAxis.add_patch(plt.Rectangle((boxT[0], boxT[1]), boxT[2] - boxT[0], boxT[3] - boxT[1],
                                            fill=False, edgecolor=colors[5], linewidth=2))
        for boxP in detectBoxes:
            if (computIOU(boxT, boxP) > 0.5): # 如果有任意一个检测框能和ground_truth_box 匹配上则TP+1
                TP += 1  # 正确检测
                break

    FN = len(true_boxes) - TP
    display_txt2 = 'TP: %i FP: %i FN: %i'%(TP, FP, FN)
    currentAxis.text(5, 15, display_txt2, bbox={'facecolor': colors[10], 'alpha': 0.5})

    TPs += TP
    FPs += FP
    FNs += FN
    plt.show()

print 'TPs: %i FPs: %i FNs: %i'%(TPs, FPs, FNs)


