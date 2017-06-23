'''
Adapted from https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py
Original author Wei Wu

Implemented the following paper:

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual Networks"
'''
import mxnet as mx
from common import conv_act_layer
from common import multibox_layer

def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tupe
        Stride used in convolution
    dim_match : Boolen
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return (conv3 + shortcut, act3)
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return (conv2 + shortcut, act2)

def resnet(units, num_stages, filter_list, num_classes, image_shape, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    """Return SSD-ResNet 101 symbol 
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_classes : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    num_unit = len(units)
    assert(num_unit == num_stages)
    label = mx.symbol.Variable(name="label")
    data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    (nchannel, height, width) = image_shape
    if height <= 32:            # such as cifar10
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                  no_bias=True, name="conv0", workspace=workspace)
    else:                       # often expected to be 224 such as imagenet
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')

    for i in range(num_stages):

        #First shortcut has different dimension in input and output, needs special treatment
        #thats controlled with dim_match=false
        (body, reluLayer) = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        #dim_match true follows.
        for j in range(units[i]-1):
            (body, reluLayer)= residual_unit(data=body, num_filter=filter_list[i+1], stride=(1,1), dim_match=True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
        if i == 2:
            #we are in block 4 (??) and attach MultiBox layer here, so remember this spot
            relu4_2 = reluLayer
    relu5_2‘ = reluLayer
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')

    ### ssd extra layers, same as in VGG atm. NOT like the google guys did ###
    conv_extra_8_1, relu_extra_8_1 = conv_act_layer(body, "8_1", 256, kernel=(1,1), pad=(0,0), \
        stride=(1,1), act_type="relu", use_batchnorm=False)
    conv_extra_8_2, relu_extra_8_2 = conv_act_layer(relu_extra_8_1, "8_2", 512, kernel=(3,3), pad=(1,1), \
        stride=(2,2), act_type="relu", use_batchnorm=False)
    conv_extra_9_1, relu_extra_9_1 = conv_act_layer(relu_extra_8_2, "9_1", 128, kernel=(1,1), pad=(0,0), \
        stride=(1,1), act_type="relu", use_batchnorm=False)
    conv_extra_9_2, relu_extra_9_2 = conv_act_layer(relu_extra_9_1, "9_2", 256, kernel=(3,3), pad=(1,1), \
        stride=(2,2), act_type="relu", use_batchnorm=False)
    conv_extra_10_1, relu_extra_10_1 = conv_act_layer(relu_extra_9_2, "10_1", 128, kernel=(1,1), pad=(0,0), \
        stride=(1,1), act_type="relu", use_batchnorm=False)
    conv_extra_10_2, relu_extra_10_2 = conv_act_layer(relu_extra_10_1, "10_2", 256, kernel=(3,3), pad=(0,0), \
        stride=(1,1), act_type="relu", use_batchnorm=False)
    conv_extra_11_1, relu_extra_11_1 = conv_act_layer(relu_extra_10_2, "11_1", 128, kernel=(1,1), pad=(0,0), \
        stride=(1,1), act_type="relu", use_batchnorm=False)
    conv_extra_11_2, relu_extra_11_2 = conv_act_layer(relu_extra_11_1, "11_2", 256, kernel=(3,3), pad=(0,0), \
        stride=(1,1), act_type="relu", use_batchnorm=False)

    from_layers = [relu4_2, relu5_2, relu_extra_8_2, relu_extra_9_2, relu_extra_10_2, relu_extra_11_2]
    sizes = [[.1, .141], [.2,.272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
    ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
        [1,2,.5], [1,2,.5]]
    normalizations = [20, -1, -1, -1, -1, -1]
    steps = [ x / 300.0 for x in [8, 16, 32, 64, 100, 300]]
    num_channels = [512]

    loc_preds, cls_preds, anchor_boxes = multibox_layer(from_layers, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_channels, clip=False, interm_layer=0, steps=steps)

    tmp = mx.contrib.symbol.MultiBoxTarget(
        *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
        ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0, \
        negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
        name="multibox_target")
    loc_target = tmp[0]
    loc_target_mask = tmp[1]
    cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
        normalization='valid', name="loc_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=0.5, force_suppress=False,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=400)
    det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, det])
    return out
    
    ### OLD HEAD OF RESNET
    # Although kernel is not used here when global_pool=True, we should put one
    #pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    #flat = mx.symbol.Flatten(data=pool1)
    #fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=num_classes, name='fc1')
    #return mx.symbol.SoftmaxOutput(data=fc1, name='softmax')

def get_symbol_train(num_classes=6, num_layers=101, image_shape=(3,300,300), conv_workspace=256, **kwargs):
    """
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    """
    (nchannel, height, width) = image_shape
    if height <= 28:
        num_stages = 3
        if (num_layers-2) % 9 == 0 and num_layers >= 164:
            per_unit = [(num_layers-2)//9]
            filter_list = [16, 64, 128, 256]
            bottle_neck = True
        elif (num_layers-2) % 6 == 0 and num_layers < 164:
            per_unit = [(num_layers-2)//6]
            filter_list = [16, 16, 32, 64]
            bottle_neck = False
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it youself".format(num_layers))
        units = per_unit * num_stages
    else:
        if num_layers >= 50:
            filter_list = [64, 256, 512, 1024, 2048]
            bottle_neck = True
        else:
            filter_list = [64, 64, 128, 256, 512]
            bottle_neck = False
        num_stages = 4
        if num_layers == 18:
            units = [2, 2, 2, 2]
        elif num_layers == 34:
            units = [3, 4, 6, 3]
        elif num_layers == 50:
            units = [3, 4, 6, 3]

        elif num_layers == 101: #we only care about this but left the rest in
            units = [3, 4, 23, 3]
        elif num_layers == 152:
            units = [3, 8, 36, 3]
        elif num_layers == 200:
            units = [3, 24, 36, 3]
        elif num_layers == 269:
            units = [3, 30, 48, 8]
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it youself".format(num_layers))

    return resnet(units       = units,
                  num_stages  = num_stages,
                  filter_list = filter_list,
                  num_classes = num_classes,
                  image_shape = image_shape,
                  bottle_neck = bottle_neck,
                  workspace   = conv_workspace)



def get_symbol(num_classes=20, nms_thresh=0.5, force_suppress=False, nms_topk=400):
    """
    Single-shot multi-box detection with VGG 16 layers ConvNet
    This is a modified version, with fc6/fc7 layers replaced by conv layers
    And the network is slightly smaller than original VGG 16 network
    This is the detection network

    Parameters:
    ----------
    num_classes: int
        number of object classes not including background
    nms_thresh : float
        threshold of overlap for non-maximum suppression
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections

    Returns:
    ----------
    mx.Symbol
    """
    net = get_symbol_train(num_classes)
    cls_preds = net.get_internals()["multibox_cls_pred_output"]
    loc_preds = net.get_internals()["multibox_loc_pred_output"]
    anchor_boxes = net.get_internals()["multibox_anchors_output"]

    cls_prob = mx.symbol.SoftmaxActivation(data=cls_preds, mode='channel', \
        name='cls_prob')
    out = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    return out