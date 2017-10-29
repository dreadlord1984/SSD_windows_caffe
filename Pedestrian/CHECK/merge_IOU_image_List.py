# -*- coding: utf-8 -*-
import numpy as np
import xml.etree.cElementTree as et
import matplotlib
import matplotlib.pyplot as plt
import linecache
import prettyplotlib as ppl

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
        fout.write((newData[0] + '\t' + newData[1] + '\t' + newData[2]))
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
def showList(IOU_all_List):
    for boxData in open(IOU_all_List).readlines():  # 对于每个box
        data = boxData.strip().split('\t')
        full_image_path = ROOTDIR + data[0]
        img = plt.imread(full_image_path)
        plt.imshow(img)
        currentAxis = plt.gca()
        # width = img.shape[1]
        # height = img.shape[0]
        full_xml_path = ROOTDIR + data[1]
        true_boxes,width,height = readXML(full_xml_path)
        print full_xml_path.decode("gb2312")

        for boxT in true_boxes:
            currentAxis.add_patch(plt.Rectangle((boxT[0], boxT[1]), boxT[2] - boxT[0], boxT[3] - boxT[1],
                                                fill=False, edgecolor=colors[5], linewidth=2))
        # 排序
        boxes_total = int(data[2])
        gt_boxes_set = set('x')
        for i in range(0, boxes_total, 1):
            gt_box_index = int(data[7 * i + 9])  # 当前匹配的gt box序号（从0开始）
            gt_boxes_set.add(gt_box_index)
        gt_boxes_set.remove('x')

        group_prior_box = [[] for x in gt_boxes_set]
        for i in range(0, boxes_total, 1):
            gt_box_index = int(data[7 * i + 9])  # 当前匹配的gt box序号（从0开始）
            databox = [float(data[7*i + 4]), float(data[7 * i + 5]) * width, float(data[7 * i + 6]) * height, float(data[7 * i + 7]) * width,
                       float(data[7 * i + 8]) * height]
            k = 0
            for gt_index in gt_boxes_set:
                if (gt_index == gt_box_index):
                    group_prior_box[k].append(databox)
                    break
                else:
                    k += 1
        for i in range(0, len(gt_boxes_set), 1):
            group_prior_box[i].sort(key=lambda x:x[0], reverse=True)

        # 显示
        for i in range(0, len(gt_boxes_set), 1):
            dispaly_box = group_prior_box[i]
            for k in range(0, len(dispaly_box), 1):
                display_txt = '%.4f' % ( dispaly_box[k][0])
                currentAxis.add_patch(plt.Rectangle((dispaly_box[k][1], dispaly_box[k][2]),
                                                    dispaly_box[k][3] - dispaly_box[k][1], dispaly_box[k][4] - dispaly_box[k][2],
                                                 fill=False, edgecolor=colors[1], linewidth=1))
                currentAxis.text(dispaly_box[k][1], dispaly_box[k][2], display_txt, bbox={'facecolor': colors[1], 'alpha': 0.5})
        plt.show()
        # break
    print "end"

"""
@function:1.绘制进入训练的prior box matching的情况
@function:2.绘制进入训练的gt box和area的情况
@param param1: 合并输出列表文件
"""
def statistic(IOU_all_List):
    with open(IOU_all_List) as f:
        for line in f:
            prior_datas = line.strip().split('\t')
            # print prior_datas[0].decode("gbk")
            prior_boxes_total = int(prior_datas[2])

            gt_set = set('x')
            for i in range(0, prior_boxes_total, 1):
                gt_box_index = (prior_datas[7 * i + 9])  # 当前匹配的gt box序号（从0开始）
                gt_set.add(gt_box_index)
            gt_set.remove('x')

            gt_group = {}
            for gt_index in gt_set:
                gt_group[gt_index] = 0

            for i in range(0, prior_boxes_total, 1):
                gt_box_index = (prior_datas[7 * i + 9])
                gt_group[gt_box_index] += 1

            for gt_index in gt_set:
                prior_box_num = gt_group[gt_index]
                for i in range(0, len(prior_nums), 1):  # 判断IOU区间段
                    if (prior_box_num <= prior_nums[i]):
                        prior_group[i] += 1
                        break
                else:
                    print prior_box_num
                    print prior_datas[0].decode("gb2312")

            full_xml_path = ROOTDIR + prior_datas[1]
            true_boxes, width, height = readXML(full_xml_path)
            for gt_index in gt_set:
                area = true_boxes[int(gt_index)][4]
                for i in range(0, len(area_thresholds), 1):  # 判断area区间段
                    if (area <= area_thresholds[i]):
                        area_group[i] += 1
                        break
                else:
                    print area


    matplotlib.rcParams['figure.figsize'] = (24, 6)  # 设定显示大小
    fig, ax = plt.subplots(1)
    labels = [prior_nums[i] for i in s_ids]
    anno_area2s = [('%d' % a) for a in prior_group[s_ids]]
    total = np.sum(prior_group)
    ppl.bar(ax, np.arange(len(prior_group)), prior_group[s_ids], annotate=anno_area2s, grid='y', xticklabels=labels)
    plt.xticks(rotation=25)
    ax.set_title('(gt total %d)' % total)
    ax.set_xlabel('matching num')
    ax.set_ylabel('gt num')
    savename = IOU_all_List[:IOU_all_List.rfind("\\")] + "\\matchNum.png"
    plt.savefig(savename)

    matplotlib.rcParams['figure.figsize'] = (8, 7)  # 设定显示大小
    fig, ax = plt.subplots(1)
    labels = [area_thresholds[i] for i in s_ids2]
    anno_area2s = [('%d' % a) for a in area_group[s_ids2]]
    total = np.sum(area_group)
    ppl.bar(ax, np.arange(len(area_group)), area_group[s_ids2], annotate=anno_area2s, grid='y', xticklabels=labels)
    plt.xticks(rotation=25)
    ax.set_title('(gt total %d)' % total)
    ax.set_xlabel('gt area')
    ax.set_ylabel('gt num')
    savename = IOU_all_List[:IOU_all_List.rfind("\\")] + "\\gtNum.png"
    plt.savefig(savename)

    plt.show()


batch_size = 32
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist() # 颜色列表
ROOTDIR = "\\\\192.168.1.186/PedestrianData/" # 样本根目录
prior_nums = np.linspace(1,40,40,dtype=np.int32)
prior_group  = np.zeros(prior_nums.size,dtype=np.int32)
area_thresholds = np.array([0.0025, 0.005, 0.01, 0.015, 0.02, 0.04, 0.08, 0.1, 0.25, 1.0],dtype=np.float64) # area 区间
area_group  = np.zeros(area_thresholds.size,dtype=np.int32)
s_ids = np.arange(prior_nums.size)
s_ids2 = np.arange(area_thresholds.size)

if __name__ == "__main__":
    # copyList("../View/COMPARE2/add_prior_gamma2_D1_new_P5N4D15E4_noSqrt/IOU_ALL.txt",
    # "../Data_0922/train_lmdb_list.txt",
    # "../View/COMPARE2/add_prior_gamma2_D1_new_P5N4D15E4_noSqrt/IOU_ALL_image_List.txt")
    # showList("../Data_0922/IOU_ALL_image_List.txt")
    statistic("..\\View\\COMPARE2\\add_prior_gamma2_D1_new_P5N4D15E4_noSqrt\\IOU_ALL_image_List.txt")