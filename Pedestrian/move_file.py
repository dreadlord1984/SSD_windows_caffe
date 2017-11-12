#coding=utf-8

import cv2
import numpy as np
import re
import lxml.etree as et
import os
import shutil


"""
@function:从xml文件中读取box信息
@param param1: xml文件
@return: boxes(with area_ratio), width, height
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
        area_ratio = float((int(xmax) - int(xmin)) * (int(ymax) - int(ymin))) / (width * height)
        boundingBox.append([int(xmin), int(ymin), int(xmax), int(ymax), area_ratio])
    return boundingBox, width, height

'''
step 1: 将验证集列表中图片和标签文件移动到目的文件夹
'''
# source_dir = '\\\\192.168.1.186\\PedestrianData\\'
# dst_dir = 'D:\\Other_Dataets\\Pedestrian\\CHECK\\'
# imge_list = 'Data_0922\\train.txt'
#
# with open(imge_list) as fp1:
#     for line in fp1:  # 每一行匹配数据 resultFile
#         img_name = source_dir + line.strip().split('.jpg ')[0] +'.jpg'
#         xml_name = source_dir + line.strip().split('.jpg ')[1]
#         dst_img_dir = dst_dir + os.path.dirname(line.strip().split('.jpg ')[0])
#         dst_xml_dir = dst_dir + os.path.dirname(line.strip().split('.jpg ')[1])
#         if not os.path.exists(img_name):
#             print 'Path does not exist: {}'.format(img_name).decode("gbk")
#         if not os.path.exists(xml_name):
#             print 'Path does not exist: {}'.format(xml_name).decode("gbk")
#         if not os.path.exists(dst_img_dir):
#             os.makedirs(dst_img_dir)
#         if not os.path.exists(dst_xml_dir):
#             os.makedirs(dst_xml_dir)
#         shutil.copy(img_name, dst_img_dir)
#         shutil.copy(xml_name, dst_xml_dir)
#
#
#         # image = cv2.imread(img_name)
#         # true_boxes, W, H = readXML(xml_name)
#         # for box in true_boxes:
#         #     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
#         #
#         # cv2.imshow('show', image)
#         # cv2.waitKey(0)

'''
step 2: 将Vedio文件夹下图片对应的ini标注文件拷贝到对应_af文件夹下
Path_Images里存的是图片从dst_dir_root之后的路径
'''
# source_dir_root = 'E:\\C++\\PedestrianFilter\\Merge\\'
# dst_dir_root = 'D:\\Other_Dataets\\Pedestrian\\CHECK\\Data_0922\\JPEGImages\\Vedio\\'
# imge_list = dst_dir_root + 'Path_Images.txt'
# with open(imge_list) as fp1:
#     for line in fp1:  # 每一行匹配数据 resultFile
#         source_dir = source_dir_root + os.path.dirname(line.strip()) + '_af'
#         if not os.path.exists(source_dir):
#             print 'Path does not exist: {}'.format(source_dir)
#         dst_dir = dst_dir_root + os.path.dirname(line.strip()) + '_af'
#         if not os.path.exists(dst_dir):
#             os.makedirs(dst_dir)
#         basename = os.path.basename(line.strip())
#         name, extension = os.path.splitext(basename)
#         source_name = os.path.join(source_dir, name + '.ini')
#         shutil.copy(source_name, dst_dir)

'''
step 3: 将其他文件夹下的xml文件转换为txt文件
删掉 'D:\Other_Dataets\Pedestrian\CHECK\Data_0922\Annotations'下的Vedio文件夹
然后cmd: dir /b/s/p/w *.xml > Path_Images.txt
替换 Path_Images.txt中 D:\Other_Dataets\Pedestrian\CHECK\Data_0922\Annotations
txt保存格式： xmin ymin xmax ymax 1.0
'''
source_dir_root = 'D:\\Other_Dataets\\Pedestrian\\CHECK\\Data_0922\Annotations\\'
dst_dir_root = 'E:\\C++\\PedestrianFilter\\Predict\\'
imge_list = 'Path_Images.txt'

with open(os.path.join(source_dir_root, imge_list)) as fp1:
    for line in fp1:  # 每一行匹配数据 resultFile
        source_xml_name = source_dir_root + line.strip()
        if not os.path.exists(source_xml_name):
            print 'Path does not exist: {}'.format(source_xml_name).decode("gbk")
        xml_name, extension = os.path.splitext(line)
        true_boxes, W, H = readXML(source_xml_name)

        # source_img_name = dst_dir_root + xml_name + '.jpg'
        # if not os.path.exists(source_img_name):
        #     print 'Path does not exist: {}'.format(source_img_name).decode("gbk")

        # image = cv2.imread(source_img_name)
        # for box in true_boxes:
        #     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
        #
        # cv2.imshow('show', image)
        # cv2.waitKey(0)

        dir_name, filename = os.path.split(xml_name)
        dst_dir = os.path.join(dst_dir_root, dir_name + '_af')
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        with open(os.path.join(dst_dir, filename+'.txt'),mode='w') as output:
            for box in true_boxes:
                output.write(str(box[0]) + ' ' + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' 1.0')
                output.write('\n')

'''step 4: 利用复查合并代码'''