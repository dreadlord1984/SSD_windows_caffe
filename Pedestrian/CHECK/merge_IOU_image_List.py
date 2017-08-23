# -*- coding: utf-8 -*-
import numpy as np
import xml.etree.cElementTree as et
import matplotlib.pyplot as plt
import linecache

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
    # width = size.find('width').text  # 子节点下节点width的值
    # height = size.find('height').text  # 子节点下节点height的值

    boundingBox = []
    for object in root.findall('object'):  # 找到root节点下的所有object节点
        bndbox = object.find('bndbox')  # 子节点下属性bndbox的值
        xmin = bndbox.find('xmin').text
        ymin = bndbox.find('ymin').text
        xmax = bndbox.find('xmax').text
        ymax = bndbox.find('ymax').text
        boundingBox.append([int(xmin), int(ymin), int(xmax), int(ymax)])
    return boundingBox

"""
@function:将匹配结果按照训练顺序和训练样本名进行合并
@param param1: 匹配列表文件
@param param2: 样本列表文件
@param param3: 待合并输出列表文件
"""
def copyList(IOUList, imageList, outList):
    imageData = open(imageList)
    lines_num = len(imageData.readlines())
    imageTmp = ''
    dataALL = []
    totalNum = 0
    for IOUdata in open(IOUList).readlines():  # 对于每个匹配数据
        datas = IOUdata.strip().split(' ')
        # if float(datas[1]) < 0.5: # 如果只统计IOU<0.5的匹配
        totalNum += 1
        num = (int(datas[7])* batch_size) + int(datas[8]) + 1 # 按照batch 索引和数据在batch中索引计算对应的样本索引
        # 由于样本总数lines_num除以batch_size可能是个小数，导致生成的IOUList里存在少量迭代两轮的样本
        # 而又因为每个batch是按照imageList顺序取得，所以如果IOUList中计算得到的索引num大于样本总数，则停止统计
        if num > lines_num :
            break

        theline = linecache.getline(imageList, num).strip().split('\t') # 对应的样本名数据
        # 因为一个样本中有多个匹配数据，我们要将属于同一个样本的匹配数据按顺序放入dataALL
        if imageTmp!=theline[0] : # 如果前后两次样本不相同
            totalNum = 1
            dataALL.append([theline[0], theline[1], str(totalNum), datas[2], datas[1], datas[3], datas[4], datas[5], datas[6], datas[9]])
            totalNum = 0
            imageTmp = theline[0]
        else:                    # 如果前后两次样本相同
            dataALL[-1][2] = str(int(dataALL[-1][2])+1)
            dataALL[-1] = dataALL[-1] + [datas[2], datas[1], datas[3], datas[4], datas[5], datas[6], datas[9]]
            imageTmp = theline[0]
        # else:
        #     continue
    fout = file(outList, "w+")
    for newData in dataALL:
        fout.write((newData[0] + '\t' + newData[1] + '\t' + newData[2]));
        for i in range(0,int(newData[2]),1):
            fout.write('\t')
            fout.write((newData[7*i + 3] + '\t' + newData[7*i + 4] + '\t' + newData[7*i + 5] + '\t'
                        + newData[7 * i + 6] + '\t' + newData[7 * i + 7]  + '\t' + newData[7 * i + 8]
                        + '\t' + newData[7 * i + 9] ))
        fout.write('\n')
    fout.close()

"""
@function:将匹配合并结果显示，红色框是ground truth box, 绿色框是 prior box
@param param1: 合并输出列表文件
"""
def showList(IOU_small_List):
    for boxData in open(IOU_small_List).readlines():  # 对于每个box
        data = boxData.strip().split('\t')
        full_image_path = ROOTDIR + data[0]
        img = plt.imread(full_image_path)
        plt.imshow(img)
        currentAxis = plt.gca()
        width = img.shape[1]
        height = img.shape[0]
        full_xml_path = ROOTDIR + data[1]
        true_boxes = readXML(full_xml_path)
        print full_xml_path.decode("gb2312")

        for boxT in true_boxes:
            currentAxis.add_patch(plt.Rectangle((boxT[0], boxT[1]), boxT[2] - boxT[0], boxT[3] - boxT[1],
                                                fill=False, edgecolor=colors[5], linewidth=2))
        # 检查下IOU_small_List中box与样本中能否对应的上
        boxes_total = int(data[2])
        for i in range(0, boxes_total, 1):
            databox_num = int(data[7*i + 9])
            display_txt = '%d %d %.2f' % ( float(data[7*i + 9]), float(data[7*i + 3]), float(data[7*i + 4]))
            databox = [float(data[7*i + 5]) * width, float(data[7*i + 6]) * height, float(data[7*i + 7]) * width, float(data[7*i + 8]) * height]
            currentAxis.add_patch(plt.Rectangle((databox[0], databox[1]), databox[2] - databox[0], databox[3] - databox[1],
                                                 fill=False, edgecolor=colors[1], linewidth=1))
            currentAxis.text(databox[0], databox[1], display_txt, bbox={'facecolor': colors[1], 'alpha': 0.5})
        plt.show()
        # break
    print "end"

batch_size = 32
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist() # 颜色列表
ROOTDIR = "\\\\192.168.1.186/PedestrianData/" # 样本根目录

if __name__ == "__main__":
    copyList("../Data_0810/IOU_ALL.txt", "../Data_0810/train_lmdb_list.txt", "../Data_0810/IOU_ALL_image_List.txt")
    # showList("../Data_0810/IOU_ALL_image_List.txt")