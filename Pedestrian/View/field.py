net_struct = {
    'train': {'net': [
                        [3, 2, 1], [3, 2, 0], [1, 1, 0], [3, 1, 1], [1, 1, 0], [3, 1, 1],
                        [2, 2, 0], [1, 1, 0], [3, 1, 1], [1, 1, 0], [3, 1, 1], [2, 2, 0],
                        [1, 1, 0], [3, 1, 1], [1, 1, 0], [3, 1, 1], [1, 1, 0], [3, 1, 1],
                        [1, 1, 0], [3, 1, 1], [1, 1, 0], [3, 2, 1], [1, 1, 0], [3, 2, 1],
                        [1, 1, 0], [3, 2, 1], [2, 2, 0]
                      ],
              'name': [
                        "conv1", "pool1", "fire2/squeeze1x1", "fire2/expand3x3", "fire3/squeeze1x1", "fire3/expand3x3",
                        "pool3", "fire4/squeeze1x1", "fire4/expand3x3", "fire5/squeeze1x1", "fire5/expand3x3", "pool5",
                        "fire6/squeeze1x1", "fire6/expand3x3", "fire7/squeeze1x1", "fire7/expand3x3", "fire8/squeeze1x1", "fire8/expand3x3",
                        "fire9/squeeze1x1", "fire9/expand3x3", "conv6_1", "conv6_2", "conv7_1", "conv7_2",
                        "conv8_1", "conv8_2", "pool6"
                       ]
              }
}
imsize = 384


def outFromIn(isz, net, layernum):
    totstride = 1
    insize = isz
    for layer in range(layernum):
        fsize, stride, pad = net[layer]
        outsize = (insize - fsize + 2 * pad) / stride + 1
        insize = outsize
        totstride = totstride * stride
    return outsize, totstride


def inFromOut(net, layernum):
    RF = 1
    for layer in reversed(range(layernum)):
        fsize, stride, pad = net[layer]
        RF = ((RF - 1) * stride) + fsize
    return RF


if __name__ == '__main__':
    print ("layer output sizes given image = %dx%d" % (imsize, imsize))

for net in net_struct.keys():
    print ('************net structrue name is %s**************' % net)
    for i in range(len(net_struct[net]['net'])):
        p = outFromIn(imsize, net_struct[net]['net'], i + 1)
        rf = inFromOut(net_struct[net]['net'], i + 1)
        print ("Layer Name = %s, Output size = %3d, Stride = % 3d, RF size = %3d" % (
        net_struct[net]['name'][i], p[0], p[1], rf))