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
thresholds = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], dtype=np.float64)
# TPs = np.array([0,    0,    1,    5,   13,   52,  137,  387,  850, 2826], dtype=np.float64)
# FNs = np.array([75,  604, 1391, 1840, 1920, 2361, 2285, 2098, 2163, 3726], dtype=np.float64)
TPs = np.zeros((len(thresholds)), dtype=np.int) # 正检
FNs = np.zeros((len(thresholds)), dtype=np.int) # 漏检
s_ids = np.arange(thresholds.size)


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
model_weights = 'snapshot_iter_120000.caffemodel'

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

ROOTDIR = "\\\\192.168.1.186/PedestrianData/"
imgList = "Data_0807/IOU_small_image_List.txt"

for imgFile in open(imgList).readlines():  # 对于每个测试图片
    plt.close('all')
    datas = imgFile.strip().split('\t')
    img_name = ROOTDIR + datas[0]
    xml_name = ROOTDIR + datas[1]
    image = caffe.io.load_image(img_name)
    width = image.shape[1]
    height = image.shape[0]
    true_boxes = readXML(xml_name) # 所有的ground truth boxes
    min_true_boxes = [] # [[IOU, [xmin, ymin, xmax, ymax]]]
    min_boxes_total = int(datas[2]) # 小匹配box数量
    for i in range(0, min_boxes_total, 1):
        min_box_num = int(datas[6 * i + 3]) # 当前小匹配box序号（从0开始）
        min_box = [float(datas[6*i + 5]) * width, float(datas[6*i + 6]) * height, float(datas[6*i + 7]) * width, float(datas[6*i + 8]) * height]
        if (computIOU(true_boxes[min_box_num], min_box) > 0.99):
            min_true_boxes.append([float(datas[6*i + 4]), true_boxes[min_box_num]])
        else:
            print "error ", img_name.decode("gb2312")
            break
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
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.5]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

    # plt.imshow(image)
    # currentAxis = plt.gca()

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
        # display_txt = '%s: %.2f'%(label_name, score)
        # coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[label]
        detectBoxes.append([xmin, ymin, xmax, ymax])
        # currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        # currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})

    for boxT in min_true_boxes:
        # currentAxis.add_patch(plt.Rectangle((boxT[1][0], boxT[1][1]), boxT[1][2] - boxT[1][0], boxT[1][3] - boxT[1][1],
        #                                     fill=False, edgecolor=colors[5], linewidth=2))
        matching = False
        div = int(boxT[0] / thresholds[0])
        for boxP in detectBoxes:
            if (computIOU(boxT[1], boxP) > 0.5): # 如果有任意一个检测框能和ground_truth_box 匹配上则TP+1
                matching = True # 正确检测
                break
        if matching:
            TPs[div] += 1
        else:
            FNs[div] += 1
    # plt.show()


print 'TPs: ', TPs
print 'FNs: ', FNs

matplotlib.rcParams['figure.figsize'] = (8, 5)  # 设定显示大小
fig, ax = plt.subplots(1)
labels = [thresholds[i] for i in s_ids]
recall2 = np.divide(TPs, np.add(TPs, FNs))
anno_area2s = [('%f' % a) for a in recall2[s_ids]]
ppl.bar(ax, np.arange(len(recall2)), recall2[s_ids], annotate=anno_area2s, grid='y', xticklabels=labels)
plt.xticks(rotation=25)
ax.set_title('(small_IOU_recall)')
ax.set_ylabel('Recall')
plt.savefig('Data_0807/small_IOU_recall.png')
plt.show()