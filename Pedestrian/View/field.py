# net_struct = {
#     'train': {'net': [
#                         [3, 2, 1], [3, 2, 0], [1, 1, 0], [3, 1, 1], [1, 1, 0], [3, 1, 1],
#                         [2, 2, 0], [1, 1, 0], [3, 1, 1], [1, 1, 0], [3, 1, 1], [2, 2, 0],
#                         [1, 1, 0], [3, 1, 1], [1, 1, 0], [3, 1, 1], [1, 1, 0], [3, 1, 1],
#                         [1, 1, 0], [3, 1, 1], [1, 1, 0], [3, 2, 1], [1, 1, 0], [3, 2, 1],
#                         [1, 1, 0], [3, 2, 1], [4, 1, 0]
#                       ],
#               'name': [
#                         "conv1", "pool1", "fire2/squeeze1x1", "fire2/expand3x3", "fire3/squeeze1x1", "fire3/expand3x3",
#                         "pool3", "fire4/squeeze1x1", "fire4/expand3x3", "fire5/squeeze1x1", "fire5/expand3x3", "pool5",
#                         "fire6/squeeze1x1", "fire6/expand3x3", "fire7/squeeze1x1", "fire7/expand3x3", "fire8/squeeze1x1", "fire8/expand3x3",
#                         "fire9/squeeze1x1", "fire9/expand3x3", "conv6_1", "conv6_2", "conv7_1", "conv7_2",
#                         "conv8_1", "conv8_2", "pool6"
#                        ]
#               }
# }
# imsize = 480
#
#
# def outFromIn(isz, net, layernum):
#     totstride = 1
#     insize = isz
#     for layer in range(layernum):
#         fsize, stride, pad = net[layer]
#         outsize = (insize - fsize + 2 * pad) / stride + 1
#         insize = outsize
#         totstride = totstride * stride
#     return outsize, totstride
#
#
# def inFromOut(net, layernum):
#     RF = 1
#     for layer in reversed(range(layernum)):
#         fsize, stride, pad = net[layer]
#         RF = ((RF - 1) * stride) + fsize
#     return RF
#
#
# if __name__ == '__main__':
#     print ("layer output sizes given image = %dx%d" % (imsize, imsize))
#
# for net in net_struct.keys():
#     print ('************net structrue name is %s**************' % net)
#     for i in range(len(net_struct[net]['net'])):
#         p = outFromIn(imsize, net_struct[net]['net'], i + 1)
#         rf = inFromOut(net_struct[net]['net'], i + 1)
#         print ("Layer Name = %s, Output size = %3d, Stride = % 3d, RF size = %3d" % (
#         net_struct[net]['name'][i], p[0], p[1], rf))

# [filter size, stride, padding]
# Assume the two dimensions are the same
# Each kernel requires the following parameters:
# - k_i: kernel size
# - s_i: stride
# - p_i: padding (if padding is uneven, right padding will higher than left padding; "SAME" option in tensorflow)
#
# Each layer i requires the following parameters to be fully represented:
# - n_i: number of feature (data layer has n_1 = imagesize )
# - j_i: distance (projected to image pixel distance) between center of two adjacent features
# - r_i: receptive field of a feature in layer i
# - start_i: position of the first feature's receptive field in layer i (idx start from 0, negative means the center fall into padding)

import math

convnet = [ [3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [2, 2, 0],
            [3, 1, 1], [3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 0],
            [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1], [2, 2, 0],
            [3, 1, 1], [1, 1, 0], [1, 1, 0], [3, 2, 1], [1, 1, 0], [3, 2, 1] ]
layer_names = [
                "conv1_1", "conv1_2", "pool1", "conv2_1", "conv2_2", "pool2",
                "conv3_1", "conv3_2", "conv3_3", "pool3", "conv4_1", "conv4_2",
                "conv4_3", "pool4", "conv5_1", "conv5_2", "conv5_3", "pool5",
                "fc6", "fc7", "conv6_1", "conv6_2", "conv7_1", "conv7_2"
            ]
imsize = 640


def outFromIn(conv, layerIn):
    n_in = layerIn[0]
    j_in = layerIn[1]
    r_in = layerIn[2]
    start_in = layerIn[3]
    k = conv[0]
    s = conv[1]
    p = conv[2]

    n_out = math.floor((n_in - k + 2 * p) / s) + 1
    actualP = (n_out - 1) * s - n_in + k
    pR = math.ceil(actualP / 2)
    pL = math.floor(actualP / 2)

    j_out = j_in * s
    r_out = r_in + (k - 1) * j_in
    start_out = start_in + ((k - 1) / 2 - pL) * j_in
    return n_out, j_out, r_out, start_out


def printLayer(layer, layer_name):
    print(layer_name + ":")
    print("\t n features: %s \n \t jump: %s \n \t receptive size: %s \t start: %s " % (
    layer[0], layer[1], layer[2], layer[3]))


layerInfos = []
if __name__ == '__main__':
    # first layer is the data layer (image) with n_0 = image size; j_0 = 1; r_0 = 1; and start_0 = 0.5
    print("-------Net summary------")
    currentLayer = [imsize, 1, 1, 0.5]
    printLayer(currentLayer, "input image")
    for i in range(len(convnet)):
        currentLayer = outFromIn(convnet[i], currentLayer)
        layerInfos.append(currentLayer)
        printLayer(currentLayer, layer_names[i])
    print("------------------------")
    layer_name = input("Layer name where the feature in: ")
    layer_idx = layer_names.index(layer_name)
    idx_x = int(input("index of the feature in x dimension (from 0)"))
    idx_y = int(input("index of the feature in y dimension (from 0)"))

    n = layerInfos[layer_idx][0]
    j = layerInfos[layer_idx][1]
    r = layerInfos[layer_idx][2]
    start = layerInfos[layer_idx][3]
    assert (idx_x < n)
    assert (idx_y < n)

    print("receptive field: (%s, %s)" % (r, r))
    print("center: (%s, %s)" % (start + idx_x * j, start + idx_y * j))