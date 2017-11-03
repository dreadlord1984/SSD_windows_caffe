# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xml.etree.cElementTree as et
import matplotlib.pyplot as plt
import prettyplotlib as ppl
import os
from matplotlib.font_manager import FontProperties  # 中文字体加载

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2

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

def boxA_in_boxB(A, B):
    if(A[0]+ 0.00001 >= B[0] and A[1] + 0.00001 >=B[1] and A[2] <= B[2] + 0.00001 and A[3] <= B[3] + 0.00001):
        return True
    else:
        return False

"""
@function:保留最大的匹配（可能有多个最大匹配）
@param param1: （去掉存在匹配大于0.5或小于0.1的gt后剩下来的）匹配结果
@return: 每个样本中目标的最大匹配
"""
def save_max_data(input_image_list, output_image_list):
    if os.path.exists(output_image_list):
        os.remove(output_image_list)
    gt_num = 0
    prior_num = 0
    for boxData in open(input_image_list).readlines():  # 对于每个box
        fout = file(output_image_list, "a")
        data = boxData.strip().split('\t')
        image_name = data[0]
        xml_name = data[1]
        fout.write(image_name)
        fout.write('\t')
        fout.write(xml_name)
        fout.write('\t')

        boxes_total = int(data[2])
        # 1. 统计gt box序号
        gt_boxes_set = set()
        for i in range(0, boxes_total, 1):
            gt_box_index = int(data[7 * i + 9])  # 当前匹配的gt box序号（从0开始）
            gt_boxes_set.add(gt_box_index)

        # 2. 将prior box按照gt box序号分组
        group_prior_box = [[] for x in gt_boxes_set]
        for i in range(0, boxes_total, 1):
            gt_box_index = int(data[7 * i + 9])  # 当前匹配的gt box序号（从0开始）
            prior_box_index = int(data[7 * i + 3])  # 当前匹配proir box序号（从0开始）
            databox = [prior_box_index, float(data[7 * i + 4]),
                       float(data[7 * i + 5]), float(data[7 * i + 6]),
                       float(data[7 * i + 7]),float(data[7 * i + 8]),
                       gt_box_index]
            k = 0
            for gt_index in gt_boxes_set:
                if (gt_index == gt_box_index):
                    group_prior_box[k].append(databox)
                    break
                else:
                    k += 1

        # 3. 排序后pop掉所有小于最大匹配IOU的prior boxes
        ####################################################################
        # # 去掉最大匹配
        # for prior_box in group_prior_box:
        #     prior_box.sort(key=lambda x: x[1], reverse=True)
        #     max_IOU = prior_box[0][1]
        #     while(1):
        #         if (len(prior_box) == 1 and prior_box[0][1] >= max_IOU - 0.000001):
        #             gt_boxes_set.remove(prior_box[0][6]) # 删除gt index 集合中
        #             group_prior_box.remove(prior_box)
        #             break
        #         elif (prior_box[0][1] >= max_IOU - 0.000001):
        #             del prior_box[0]
        #         else:
        #             break
        ####################################################################
        gt_num += len(gt_boxes_set)
        for prior_box in group_prior_box:
            prior_box.sort(key=lambda x: x[1], reverse=True)
            max_IOU = prior_box[0][1]
            for k in range(1, len(prior_box), 1): #去掉所有小于max_IOU的匹配
                if (prior_box[-1][1] < max_IOU - 0.000001):
                    prior_box.pop()
                else:
                    break

        total = 0
        for one_target_prior_boxes in group_prior_box:
            total += len(one_target_prior_boxes)
        fout.write(str(total))

        # 4. 保存最大匹配
        for one_target_prior_boxes in group_prior_box:
            prior_num += len(one_target_prior_boxes)
            for prior_box in one_target_prior_boxes:
                fout.write('\t')
                fout.write(str(prior_box[0]) +  '\t' + str(prior_box[1]) +  '\t'
                           + str(prior_box[2]) +  '\t' + str(prior_box[3]) +  '\t'
                           + str(prior_box[4]) + '\t' +  str(prior_box[5]) +  '\t'
                           + str(prior_box[6]))
        fout.write('\n')

        fout.close()
    print "gt_num: %d" % gt_num
    print "prior_num: %d" % prior_num

"""
@function:一个样本一行，统计（不）完全包含gt box的匹配
@param param1: 匹配结果列表
@return: 输入的匹配结果中prior box完全包含gt box的匹配 
"""
def save_include_data(input_image_list, output_include_list, output_uninclude_list):
    with open(input_image_list) as fp1:
        fout = file(output_include_list, "w+")
        fout2 = file(output_uninclude_list, "w+")
        include_prior_num = 0 # 完全被包含的gt box的prior box数量
        uninclude_prior_num = 0 # 部分包含的gt box的prior box数量
        include_gt_num = 0 # 完全被包含的gt box数量
        uninclude_gt_num = 0 # 部分被包含的gt box数量

        for imageFile in fp1:  # 每一行匹配数据 resultFile
            prior_datas = imageFile.strip().split('\t')
            img_name = prior_datas[0]
            xml_name = prior_datas[1]
            # 1. 读取gt box数量
            true_boxes, width, height = readXML(ROOTDIR + xml_name)  # 所有的ground truth boxes

            # 2. 读取prior box, 按照是否完全包含gt box分成两类include_prior_boxes、uninclude_prior_boxes
            include_prior_boxes = []
            uninclude_prior_boxes = []
            prior_boxes_total = int(prior_datas[2])

            gt_index_set = set()
            for i in range(0, prior_boxes_total, 1):
                prior_box_index = int(prior_datas[7 * i + 3])  # 当前匹配proir box序号（从0开始）
                prior_box_IOU = prior_datas[7 * i + 4] # 当前匹配proir box序号（从0开始）
                prior_box_coordinates = [round(float(prior_datas[7 * i + 5]) * width), round(float(prior_datas[7 * i + 6]) * height),
                                         round(float(prior_datas[7 * i + 7]) * width), round(float(prior_datas[7 * i + 8]) * height)]
                gt_box_index = int(prior_datas[7 * i + 9])  # 当前匹配的gt box序号（从0开始）
                if(boxA_in_boxB(true_boxes[gt_box_index], prior_box_coordinates)):
                    gt_index_set.add(gt_box_index)
                    include_prior_boxes.append([prior_box_index, prior_box_IOU, prior_box_coordinates, gt_box_index])
                else:
                    uninclude_prior_boxes.append([prior_box_index, prior_box_IOU, prior_box_coordinates, gt_box_index])

            # 3. 统计输出完全包含gt box的include_prior_boxes
            include_gt_boxes_set = set()
            if (len(include_prior_boxes) > 0):
                include_prior_num += len(include_prior_boxes)
                fout.write(( img_name + '\t' + xml_name + '\t' + str(len(include_prior_boxes)) ))
                for k in range(0, len(include_prior_boxes), 1):
                    include_gt_boxes_set.add(str(include_prior_boxes[k][3]))
                    fout.write('\t')
                    fout.write(( str(include_prior_boxes[k][0]) + '\t' + str(include_prior_boxes[k][1]) + '\t'
                                + str(include_prior_boxes[k][2][0]) + '\t' + str(include_prior_boxes[k][2][1]) + '\t'
                                + str(include_prior_boxes[k][2][2]) + '\t' + str(include_prior_boxes[k][2][3]) + '\t'
                                + str(include_prior_boxes[k][3]) ))
                fout.write('\n')

            include_gt_num += len(include_gt_boxes_set)
            # 4. 统计部分包含gt box的uninclude_prior_boxes
            uninclude_gt_boxes_set = set()
            if (len(uninclude_prior_boxes) > 0):
                uninclude_prior_num += len(uninclude_prior_boxes)
                fout2.write((img_name + '\t' + xml_name + '\t' + str(len(uninclude_prior_boxes))))
                for k in range(0, len(uninclude_prior_boxes), 1):
                    uninclude_gt_boxes_set.add(str(uninclude_prior_boxes[k][3]))
                    fout2.write('\t')
                    fout2.write((str(uninclude_prior_boxes[k][0]) + '\t' + str(uninclude_prior_boxes[k][1]) + '\t'
                                + str(uninclude_prior_boxes[k][2][0]) + '\t' + str(uninclude_prior_boxes[k][2][1]) + '\t'
                                + str(uninclude_prior_boxes[k][2][2]) + '\t' + str(uninclude_prior_boxes[k][2][3]) + '\t'
                                + str(uninclude_prior_boxes[k][3])))
                fout2.write('\n')
            uninclude_gt_num += len(uninclude_gt_boxes_set)

        fout.close()
        fout2.close()

        print "include_prior_num: %d" % include_prior_num
        print "uninclude_prior_num: %d" % uninclude_prior_num
        print "include_gt_num: %d" % include_gt_num
        print "uninclude_gt_num: %d" % uninclude_gt_num

"""
@function:模型测试结果
@param param1: 匹配列表
@return: NONE
"""
def test_list(imgList, result_file):
    net = caffe.Net(model_def,  # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)  # use test mode (e.g., don't perform dropout)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

    # set net to batch size of 1
    resize_width = 384
    resize_height = 256
    net.blobs['data'].reshape(1, 3, resize_height, resize_width)
    if os.path.exists(result_file):
        os.remove(result_file)
    for imgFile in open(imgList).readlines():  # 对于每个测试图片
        img_name = ROOTDIR + imgFile.strip().split('\t')[0]
        output = open(result_file, 'a')
        output.write(imgFile.strip().split('\t')[0])
        image = caffe.io.load_image(img_name)
        # true_boxes = readXML(xml_name);
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image
        # Forward pass.
        detections = net.forward()['detection_out']

        # Parse the outputs.
        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3]
        det_ymin = detections[0, 0, :, 4]
        det_xmax = detections[0, 0, :, 5]
        det_ymax = detections[0, 0, :, 6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.05]

        top_conf = det_conf[top_indices]
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        output.write('\t')
        output.write(str(top_conf.shape[0]))
        for i in xrange(top_conf.shape[0]):  # 对每个检测到的目标
            xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            output.write('\t')
            output.write(str(score))
            output.write('\t')
            output.write(str(xmin))
            output.write('\t')
            output.write(str(ymin))
            output.write('\t')
            output.write(str(xmax))
            output.write('\t')
            output.write(str(ymax))

        output.write('\n')
        output.close()

"""
@function:一个目标一行，统计来自底层的完全包含gt box的匹配的IOU
@param param1: 匹配结果中prior box完全包含gt box的匹配 
@return: 每个目标中来自底层的匹配 
"""
def save_lower_layer_prior_box(input_include_list, output_layer_prior_box):
    with open(input_include_list) as fp1:
        fout = file(output_layer_prior_box, "w+")
        for imageFile in fp1:  # 每一行匹配数据 resultFile
            prior_datas = imageFile.strip().split('\t')
            prior_boxes_total = int(prior_datas[2])
            num = 0
            old_gt_index = -1
            for i in range(0, prior_boxes_total, 1):
                prior_box_index = int(prior_datas[7 * i + 3])  # 当前匹配proir box序号（从0开始）
                gt_box_index = int(prior_datas[7 * i + 9])  # 当前匹配的gt box序号（从0开始）
                if(prior_box_index < layer_priorbox_num[0]):
                    if(num == 0): # 一张图片重头开始
                        num = 1
                        old_gt_index = gt_box_index
                        fout.write(prior_datas[7 * i + 4])
                    elif(gt_box_index != old_gt_index): # gt_box序号发生变化
                        num = 1
                        old_gt_index = gt_box_index
                        fout.write('\n')
                        fout.write(prior_datas[7 * i + 4])
                    else:
                        fout.write('\t')
                        fout.write(prior_datas[7 * i + 4])
            if(num != 0):
                fout.write('\n')
        fout.close()

"""
@function:统计显示底层中完全包含gt box的area
@param param1: 匹配结果中prior box完全包含gt box的匹配
@return: NONE
"""
def show_lower_layer_gt_area(input_include_list):
    fig, axes = plt.subplots(nrows=1, figsize=(8, 8))
    num = 0
    with open(input_include_list) as fp1:
        for imageFile in fp1:  # 每一行匹配数据 resultFile
            prior_datas = imageFile.strip().split('\t')
            img_name = prior_datas[0]
            xml_name = prior_datas[1]
            # 1. 读取gt box数量
            true_boxes, width, height = readXML(ROOTDIR + xml_name)  # 所有的ground truth boxes

            # 2. 记录被底层prior box完全包含的gt box序号
            prior_boxes_total = int(prior_datas[2])
            include_gt_boxes_set_in = set()
            include_gt_boxes_set_out = set()
            for i in range(0, prior_boxes_total, 1):
                prior_box_index = int(prior_datas[7 * i + 3])  # 当前匹配proir box序号（从0开始）
                gt_box_index = int(prior_datas[7 * i + 9])  # 当前匹配的gt box序号（从0开始）
                if (prior_box_index < layer_priorbox_num[0]):
                    include_gt_boxes_set_in.add(gt_box_index)
                else:
                    include_gt_boxes_set_out.add(gt_box_index)

            # include_gt_boxes_set = include_gt_boxes_set_out - include_gt_boxes_set_in # 从非底层中去掉来自底层

            # 3. 统计被包含的gt box面积分布
            for gt_index in include_gt_boxes_set_out:
                num += 1
                gt_box = true_boxes[gt_index]
                area_ratio = float((gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])) / (width * height)
                for i in range(0, len(area_thresholds), 1):  # 判断ratio区间段
                    if (area_ratio <= area_thresholds[i]):
                        area_gt_num[i] += 1
                        break

    labels = [area_thresholds[i] for i in s_ids2]
    ppl.bar(axes, s_ids2, area_gt_num,
            annotate=True,
            grid='y', xticklabels=labels,
            color=[colors[i] for i in s_ids2])
    axes.set_title("uninclude GT area out, total " + str(num))
    axes.set_ylabel("NUM")
    axes.set_xlabel('gt box area ratio')
    plt.show()

"""
@function:显示完全被底层prior box包含的gt box的最大IOU的分布情况
@param param1: 从大到小排序后，每个目标中来着底层的匹配
@return: NONE
"""
def show_max_lower_layer_prior_box(input):
    fig, axes = plt.subplots(nrows=1, figsize=(8, 8))
    num = 0
    with open(input) as fp1:
        for gt_all_prior in fp1:  # 每一行匹配数据 resultFile
            max_IOU = float(gt_all_prior.strip().split('\t')[0])
            num += 1
            for i in range(0, len(max_IOU_area), 1):  # 判断ratio区间段
                if (max_IOU <= max_IOU_area[i]):
                    area_prior_num[i] += 1
                    break

    labels = [max_IOU_area[i] for i in s_ids]
    ppl.bar(axes, s_ids, area_prior_num,
            annotate=True,
            grid='y', xticklabels=labels,
            color=[colors[i] for i in s_ids])
    axes.set_title("MAX IOU, total " + str(num))
    axes.set_ylabel("NUM")
    axes.set_xlabel('IOU ratio')
    plt.show()

"""
@function:读取excel表绘制统计
@param param1: 表格文件
@return: NONE
"""
def statistical_analysis(input_excel):
    # 1. 按max IOU对所有gt box进行划分显示
    gt_data = pd.DataFrame(pd.read_excel(input_excel, 'Sheet1'))
    gt_grade = gt_data.groupby('grade')['gt_num'].agg(sum)
    # 图表字体为华文细黑，字号为15，窗口大小
    # plt.rc('font', family='STXihei', size=14)
    plt.figure(figsize=(8, 8))
    # 设置饼图中每个数据分类的颜色
    colors = ["DeepPink", "lightskyblue", "Yellow"]
    # 设置饼图中每个数据分类的名称
    name = [u'0.1<=maxIOU<0.5', u'maxIOU<0.1', u'maxIOU>=0.5']
    # 创建饼图，设置分类标签，颜色和图表起始位置等
    plt.pie(gt_grade, labels=name,
            colors=colors,# 设置饼图的自定义填充色
            textprops = { 'fontsize': 15, 'color': 'k'}, # 设置文本标签的属性值
            wedgeprops = { 'linewidth': 1, 'edgecolor': 'White'}, # 设置饼图内外边界的属性值
            explode=(0.05, 0, 0), # 突出显示
            startangle=120, # 设置饼图的初始角度
            radius=1,  # 设置饼图的半径
            labeldistance=1,  # 设置水平标签与圆心的距离
            pctdistance=0.6,  # 设置百分比标签与圆心的距离
            autopct='%1.1f%%', #设置百分比的格式，这里保留一位小数
            )
    # 添加图表标题
    tmp = gt_grade._values.tolist()
    legend_name = []
    for a, b in zip(name, tmp):
        legend_name.append(a+" (" + str(b) + ")")
    plt.title(u'按照最大匹配IOU划分ground truth, sum=%d' %np.sum(gt_grade._values), fontproperties=font)
    # 添加图例，并设置显示位置
    plt.legend(legend_name, loc='upper right')


    # 2. 将上面0.1<=maxIOU<0.5部分的gt box按照是否被prior box完全包含进行划分显示
    # 显示gt box
    fig, axes = plt.subplots(ncols=2, figsize=(16, 8))
    gt_data2 = pd.DataFrame(pd.read_excel(input_excel, 'Sheet2'))
    gt_grade2 = gt_data2.groupby('status')['gt_num'].agg(sum)
    name2 = [u'include gt', u'uninclude gt']
    axes[0].pie(gt_grade2, labels=name2,
            colors=colors,# 设置饼图的自定义填充色
            textprops = { 'fontsize': 15, 'color': 'k'}, # 设置文本标签的属性值
            wedgeprops = { 'linewidth': 1, 'edgecolor': 'White'}, # 设置饼图内外边界的属性值
            explode=(0.05, 0), # 突出显示
            startangle=120, # 设置饼图的初始角度
            radius=1,  # 设置饼图的半径
            labeldistance=1,  # 设置水平标签与圆心的距离
            pctdistance=0.6,  # 设置百分比标签与圆心的距离
            autopct='%1.1f%%', #设置百分比的格式，这里保留一位小数
            )
    # 添加图表标题
    tmp = gt_grade2._values.tolist()
    legend_name2 = []
    for a, b in zip(name2, tmp):
        legend_name2.append(a+" (" + str(b) + ")")
    axes[0].set_title(u'按照是否被prior box完全包含住划分ground truth, gt sum=%d' % np.sum(gt_grade2._values), fontproperties=font)
    axes[0].legend(legend_name2, loc='lower right')

    # 显示prior box
    gt_grade3 = gt_data2.groupby('status')['prior_num'].agg(sum)
    name3 = [u'prior include', u'prior uninclude']
    axes[1].pie(gt_grade3, labels=name2,
                colors=colors,  # 设置饼图的自定义填充色
                textprops={'fontsize': 15, 'color': 'k'},  # 设置文本标签的属性值
                wedgeprops={'linewidth': 1, 'edgecolor': 'White'},  # 设置饼图内外边界的属性值
                explode=(0.05, 0),  # 突出显示
                startangle=120,  # 设置饼图的初始角度
                radius=1,  # 设置饼图的半径
                labeldistance=1,  # 设置水平标签与圆心的距离
                pctdistance=0.6,  # 设置百分比标签与圆心的距离
                autopct='%1.1f%%',  # 设置百分比的格式，这里保留一位小数
                )
    # 添加图表标题
    tmp = gt_grade3._values.tolist()
    legend_name3 = []
    for a, b in zip(name3, tmp):
        legend_name3.append(a + " (" + str(b) + ")")
    axes[1].set_title(u'按照上面划分后对应prior box分布, prior sum=%d' % np.sum(gt_grade3._values), fontproperties=font)
    axes[1].legend(legend_name3, loc='lower right')

    plt.show()

ROOTDIR = "\\\\192.168.1.186/PedestrianData/" # 样本所在根目录
layer_priorbox_num = np.array([9216, 2304, 576, 144, 36, 6],dtype=np.int32) # layer层priorbox 数
max_IOU_area = np.array([0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5],dtype=np.float32) # layer层priorbox 数
area_prior_num = np.zeros(len(max_IOU_area),dtype=np.int32)
s_ids = np.arange(len(max_IOU_area))
area_thresholds = np.array([0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.01, 0.015, 0.02, 0.04, 0.08, 0.1, 0.25, 1.0],dtype=np.float64) # area 区间
area_gt_num = np.zeros(len(area_thresholds),dtype=np.int32)
s_ids2 = np.arange(len(area_thresholds))
colors = plt.cm.hsv(np.linspace(0, 1, 18)).tolist()
font = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\msyh.ttc", size=14)

# load PASCAL VOC labels
labelmap_file = 'labelmap_VehicleFull.prototxt'
filelabel = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(filelabel.read()), labelmap)

model_def = '../deploy2.prototxt' # 检测网络
model_weights = '..\\View\\COMPARE\\NONE_A75G20_S_D\\NONE_A75G20_S_D_fix_n_iter_200000.caffemodel' # 训练好的模型
ROOTDIR = "\\\\192.168.1.186\\PedestrianData\\" # 待测试样本集所在根目录
imgList = "..\\Data_0825\\val.txt" # 样本列表

if __name__ == "__main__":
    # 1. 对于去掉存在大于0.5和小于0.1的匹配后剩下的样本中，只保留最大的匹配（可能有多个最大匹配）
    # save_max_data("../Data_0922/eliminate_greater0.5_less0.1_image_list.txt",
    #           "../Data_0922/eliminate_greater0.5_less0.1_max_image_list.txt")

    # 2. 对于只保留的最大匹配后剩下的样本中，一个样本一行，统计完全和不完全包含gt box的匹配
    # save_include_data("../Data_0922/eliminate_greater0.5_less0.1_max_image_list.txt",
    #           "../Data_0922/include_gt.txt", "../Data_0922/uninclude_gt.txt")

    # 3. 测试对于去掉存在大于0.5和小于0.1的匹配后剩下的样本的最大匹配
    # test_list("../Data_0922/eliminate_greater0.5_less0.1_max_image_list.txt",
    #           "../Data_0922/eliminate_greater0.5_less0.1_detect_result.txt")

    # 4. 一个目标一行，统计来自底层（高层higher_layer_prior_box >=）中完全包含gt box的匹配的IOU
    # save_lower_layer_prior_box("../Data_0922/include_gt.txt",
    #            "../Data_0922/lower_layer_prior_box.txt")

    # 5. 显示完全被底层prior box包含的gt box的最大IOU的分布情况
    # show_max_lower_layer_prior_box("../Data_0922/lower_layer_prior_box.txt")

    # 6. 统计显示底层（非底层）中完全包含gt box的area
    # show_lower_layer_gt_area("../Data_0922/uninclude_gt.txt")

    # 7. 读取excel表绘制统计
    statistical_analysis('../Data_0922/prior_data.xlsx')

