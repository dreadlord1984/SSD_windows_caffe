# -*- coding: utf-8 -*-
import numpy as np
import xml.etree.cElementTree as et
import matplotlib.pyplot as plt
import scipy.io
import os
import math
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
def statistic(IOU_all_List, save_name):
    mat_save_name = os.path.join(os.path.dirname(IOU_all_List), save_name)
    with open(IOU_all_List) as data:
        for line in data:
            prior_datas = line.strip().split('\t')
            prior_boxes_total = int(prior_datas[2])
            full_xml_path = ROOTDIR + prior_datas[1]
            true_boxes, width, height = readXML(full_xml_path)

            # 1. 将 gt box 按照gt box height分组
            for gt_index in range(0, len(true_boxes), 1):
                index = int((true_boxes[gt_index][3] - true_boxes[gt_index][1]) * 320 / height) # 往小取整！
                if index <= 9:
                    print prior_datas[0].decode("gbk") + ' ' +  str(index) + ' '+ str(true_boxes[gt_index][3])
                    shutil.copy(ROOTDIR + prior_datas[0], bad_image_root)
                    shutil.copy(ROOTDIR + prior_datas[1], bad_xml_root)
                gt_height_group[index] += 1
            # 2. 将 priot box 按照匹配的gt box height分组
            for i in range(0, prior_boxes_total, 1):
                if int(prior_datas[7 * i + 3]) == -1: # prior box index
                    no_match_index = int((float(prior_datas[7 * i + 8]) - float(prior_datas[7 * i + 6])) * 320) # 往小取整！
                    no_match_gt_height_group[no_match_index] += 1
                    continue
                else:
                    gt_box_index = int(prior_datas[7 * i + 9])  # 当前匹配的gt box序号（从0开始）
                    try:
                        index = int((true_boxes[gt_box_index][3] - true_boxes[gt_box_index][1]) * 320 / height)
                        if index >= height_come_from_min and index <= height_come_from_max:
                            prior_box_index = int(prior_datas[7 * i + 3])
                            for k in range(0, len(layer_thresholds), 1):  # 判断来自于哪一层
                                if (prior_box_index < layer_thresholds[k]):
                                    layer_group[k] += 1
                                    break
                        prior_gt_group[index] += 1
                    except:
                        print 'error occurs while reading gt box'

    scipy.io.savemat(mat_save_name, {'gt_height_group': gt_height_group,
                                     'prior_gt_group':prior_gt_group,
                                     'no_match_gt_height_group': no_match_gt_height_group})


def draw_curve(*curves):
    plt.rcParams['figure.figsize'] = (9, 10)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    fig, axes = plt.subplots(nrows=2)
    fig2, axes2 = plt.subplots(nrows=1, figsize=(9, 10))
    for curve_i, curve_name in enumerate(curves):
        # fig, axes = plt.subplots(nrows=2)
        data = scipy.io.loadmat(curve_name)
        (ini_root, filename) = os.path.split(curve_name)
        filename, extension = os.path.splitext(filename)
        gt_height_group_data = data['gt_height_group'][0]
        prior_gt_group_data = data['prior_gt_group'][0]
        no_match_gt_height_group = data['no_match_gt_height_group'][0]
        ##########################################################################
        for index in range(0, len(gt_height_group_data), 1):  # 求平均
            if gt_height_group_data[index] != 0:
                mean_anchor_group[index] = float(prior_gt_group_data[index]) / gt_height_group_data[index]
        axes[0].plot(mean_anchor_group, color=colors[curve_i], linewidth=1, markeredgewidth=3,
                 markeredgecolor='#99CC01', alpha=0.8, label=filename)
        # 添加x轴标签
        axes[0].set_xlabel('Sacle of pedestrian')
        # 添加y周标签
        axes[0].set_ylabel(u'Number of matched anchors')
        # 添加图表标题
        axes[0].set_title(u'anchor-gt distribute')
        # total_0 = np.sum(mean_anchor_group)
        # axes[0].set_title(u'anchor-gt distribute: {}'.format(total_0))
        # 添加图表网格线，设置网格线颜色，线形，宽度和透明度
        plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.4)
        # axes[0].legend(loc="lower left")
        ##########################################################################
        axes[1].plot(gt_height_group_data, color=colors[curve_i], linewidth=1, markeredgewidth=3,
                 markeredgecolor='#99CC01', alpha=0.8, label=filename )
        # 添加x轴标签
        axes[1].set_xlabel('Sacle of pedestrian')
        # 添加y周标签
        axes[1].set_ylabel(u'Number of gt')
        # 添加图表标题
        # axes[1].set_title(u'gt distribute')
        total_1 = np.sum(gt_height_group_data)
        axes[1].set_title(u'gt distribute: {}'.format(total_1))
        axes[1].legend(loc="lower left")
        ##########################################################################
        total_2 = np.sum(no_match_gt_height_group)
        max_indx = np.argmax(no_match_gt_height_group)
        axes2.plot(max_indx ,no_match_gt_height_group[max_indx], 's', color=colors[curve_i])
        show_max = '[' + str(max_indx) + ' ' + str(no_match_gt_height_group[max_indx]) + ']'
        axes2.annotate(show_max, xytext=(max_indx, no_match_gt_height_group[max_indx]), xy=(max_indx, no_match_gt_height_group[max_indx]))
        axes2.plot(no_match_gt_height_group, color=colors[curve_i], linewidth=1, markeredgewidth=3,
                markeredgecolor='#99CC01', alpha=0.8, label=filename+'({})'.format(total_2))
        axes2.set_title('no match gt box')
        axes2.set_xlabel('Sacle of pedestrian')
        axes2.set_ylabel(u'Number of gt')

        # 添加图表网格线，设置网格线颜色，线形，宽度和透明度
        plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.4)
        # 设置数据分类名称
        # plt.xticks(a, (u'1月', u'2月', u'3月', u'4月', u'5月', u'6月', u'7月', u'8月', u'9月', u'10月', u'11月', u'12月'))
        axes2.legend(loc="upper right")
    # 输出图表
    plt.show()



ROOTDIR = "\\\\192.168.1.186/PedestrianData/" # 样本根目录
gt_heights = np.linspace(1,320,320,dtype=np.int32)
gt_height_group  = np.zeros(gt_heights.size,dtype=np.int32)
prior_gt_group  = np.zeros(gt_heights.size,dtype=np.int32)
mean_anchor_group = np.zeros(gt_heights.size,dtype=np.float)
layer_thresholds = np.array([12000, 24000, 27000, 27750, 27950, 28010, 28015],dtype=np.int32) # layer层priorbox 数
layer_group = np.zeros(layer_thresholds.size,dtype=np.int32)
height_come_from_min = 74
height_come_from_max = 84
no_match_gt_height_group  = np.zeros(gt_heights.size,dtype=np.int32)
bad_image_root = 'D:\\Other_Dataets\\Pedestrian\\view\\JPEGImages'
bad_xml_root = 'D:\\Other_Dataets\\Pedestrian\\view\\Annotations'
colors = ['Blue', 'Red',  'MediumAquamarine', 'HotPink', 'Chartreuse','Gray', 'Chocolate']

SRCDIR = u'E:\\caffe-master_\\Pedestrian\\Data_480_320\\19,32_64_118_172_227_281'.encode('gbk')
if __name__ == "__main__":
    # statistic(os.path.join(SRCDIR, "IOU_ALL_VAL_NEW_48_IOU45_image_list.txt"),"Anchor_Group_NEW_48_IOU45")
    draw_curve(
        # os.path.join(SRCDIR, "Anchor_Group.mat"),
        # os.path.join(SRCDIR, "VAL_Anchor_Group.mat"),
        # os.path.join(SRCDIR, "Anchor_Group_NEW.mat"),
        os.path.join(SRCDIR, "VAL_Anchor_Group_NEW.mat"),
        # os.path.join(SRCDIR, "VAL_Anchor_Group_NEW_0.25_0.33_0.5_0.75_ALL.mat"),
        # os.path.join(SRCDIR, "VAL_Anchor_Group_NEW_0.2_0.33_0.5_0.75_ALL.mat"),
        # os.path.join(SRCDIR, "VAL_Anchor_Group_NEW_0.2_0.25_0.33_0.5_0.75_ALL.mat"),
        os.path.join(SRCDIR, "VAL_Anchor_Group_NEW_48.mat"),
        os.path.join(SRCDIR, "Anchor_Group_NEW_48_IOU45.mat"),
               )
