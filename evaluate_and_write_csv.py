import argparse
import glob
import importlib
import os
import sys
import numpy as np
import pandas as pd
import mxnet as mx

from detect.detector import Detector

from evaluate.eval_false_positive import  evaluate_false_positives


def get_detector(net, prefix, epoch, data_shape, mean_pixels, ctx,
                 classes,
                 nms_thresh=0.5, force_nms=True):
    """
    wrapper for initialize a detector

    Parameters:
    ----------
    net : str
        test network name
    prefix : str
        load model prefix
    epoch : int
        load model epoch
    data_shape : int
        resize image shape
    mean_pixels : tuple (float, float, float)
        mean pixel values (R, G, B)
    ctx : mx.ctx
        running context, mx.cpu() or mx.gpu(?)
    force_nms : bool
        force suppress different categories
    """
    sys.path.append(os.path.join(os.getcwd(), 'symbol'))
    if net is not None:
        net = importlib.import_module("symbol_" + net) \
            .get_symbol(len(classes), nms_thresh, force_nms)
    detector = Detector(net, prefix + "_" + str(data_shape), epoch, \
                        data_shape, mean_pixels, ctx=ctx)
    return detector

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a network')
    parser.add_argument('--rec-path', dest='rec_path', help='which record file to use',
                        default=os.path.join(os.getcwd(), 'data', 'val.rec'), type=str)
    parser.add_argument('--list-path', dest='list_path', help='which list file to use',
                        default="", type=str)
    parser.add_argument('--network', dest='network', type=str, default='vgg16_ssd_300',
                        choices=['vgg16_ssd_300', 'vgg16_ssd_512'], help='which network to use')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32,
                        help='evaluation batch size')
    parser.add_argument('--num-class', dest='num_class', type=int, default=20,
                        help='number of classes')
    parser.add_argument('--class-names', dest='class_names', type=str, default="",
                        help='string of comma separated names, or text filename')
    parser.add_argument('--epoch', dest='epoch', help='epoch of pretrained model',
                        default=0, type=int)
    parser.add_argument('--prefix', dest='prefix', help='load model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'ssd'), type=str)
    parser.add_argument('--gpus', dest='gpu_id', help='GPU devices to evaluate with',
                        default='0', type=str)
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=300,
                        help='set image shape')
    parser.add_argument('--mean-r', dest='mean_r', type=float, default=123,
                        help='red mean value')
    parser.add_argument('--mean-g', dest='mean_g', type=float, default=117,
                        help='green mean value')
    parser.add_argument('--mean-b', dest='mean_b', type=float, default=104,
                        help='blue mean value')
    parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.45,
                        help='non-maximum suppression threshold')
    parser.add_argument('--overlap', dest='overlap_thresh', type=float, default=0.5,
                        help='evaluation overlap threshold')
    parser.add_argument('--force', dest='force_nms', type=bool, default=False,
                        help='force non-maximum suppression on different class')
    parser.add_argument('--use-difficult', dest='use_difficult', type=bool, default=False,
                        help='use difficult ground-truths in evaluation')
    parser.add_argument('--voc07', dest='use_voc07_metric', type=bool, default=True,
                        help='use PASCAL VOC 07 metric')
    parser.add_argument('--deploy', dest='deploy_net', help='Load network from model',
                        action='store_true', default=False)
    parser.add_argument('--use-fp-metric', dest='use_fp', default=True, type=bool, help='Evaluate false positives as well')
    parser.add_argument('--false-positives-image-folder', dest='fp_image_folder', type=str,
                        help='Images to be used for false postive metric', default='')
    parser.add_argument('--false-positives-thresholds', dest='fp_threshold', type=str,
                        help='Threshold for counting a detection for false positives', default='0.5,0.75')
    parser.add_argument('--false-postives-image-count', dest='fp_img_count', type=int,
                        help='Number of images to be used for fp detection', default=500)
    parser.add_argument('--cpu', dest='cpu', help='(override GPU) use CPU to detect',
                        action='store_true', default=False)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.cpu:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(int(args.gpu_id))

    image_glob = os.path.join(args.fp_image_folder, '*.jpg')
    image_list = glob.glob(image_glob)
    if len(image_list) > args.fp_img_count:
        image_list = image_list[:args.fp_img_count]
    print "Going to process", len(image_list), "images"
    assert len(image_list) > 0, "No valid image specified to detect"
    network = args.network
    detector = get_detector(network, args.prefix, args.epoch,
                            args.data_shape,
                            (args.mean_r, args.mean_g, args.mean_b),
                            ctx,
                            args.class_names.split(","),
                            args.nms_thresh, args.force_nms)
    thresholds = [float(str) for str in args.fp_threshold.split(',')]
    false_positives = evaluate_false_positives(detector, image_list, args.class_names.split(","), thresholds)

    scores_df = pd.read_csv(args.prefix+'.csv', index_col=0)

    for threshold,data in false_positives.iteritems():
        for label,score in data.iteritems():
            fp_tag = 'fp_%s_%2.2f' % (label,threshold)
            if fp_tag not in scores_df:
                scores_df[fp_tag] = np.nan

            scores_df.loc[args.epoch, fp_tag] = score
    scores_df.fillna(0, inplace=True)

    scores_df.to_csv(args.prefix+'.csv')
    #print(false_positives)