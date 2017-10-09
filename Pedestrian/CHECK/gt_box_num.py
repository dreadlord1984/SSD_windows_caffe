# -*- coding: utf-8 -*-
import numpy as np
import xml.etree.cElementTree as et

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

# ROOTDIR = "\\\\192.168.1.186/PedestrianData/" # 样本所在根目录
# with open("../Data_0922/train_lmdb_list.txt") as fp1:
#     box_num = 0
#     for imageFile in fp1:  # 每一行匹配数据 resultFile
#         image_datas = imageFile.strip().split('\t')
#         xml_name = ROOTDIR + image_datas[1]
#         # 1. 统计gt box数量
#         true_boxes, width, height = readXML(xml_name)  # 所有的ground truth boxes
#         box_num += len(true_boxes)
#     print box_num

with open("../Data_0922/gt_num1.txt") as fp1:
    box_num = 0
    for imageFile in fp1:  # 每一行匹配数据 resultFile
        tmp = int(imageFile)
        box_num += tmp
    print box_num

with open("../Data_0922/gt_num2.txt") as fp1:
    box_num = 0
    for imageFile in fp1:  # 每一行匹配数据 resultFile
        tmp = int(imageFile)
        box_num += tmp
    print box_num