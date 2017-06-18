import argparse
import glob
import json

import math

import cv2

import tools.find_mxnet
import mxnet as mx
import os
import importlib
import sys
from detect.detector import Detector

CLASSES = ("berlinerdom", "brandenburgertor", "rathausberlin", "reichstag")


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
            .get_symbol(len(CLASSES), nms_thresh, force_nms)
    detector = Detector(net, prefix + "berlin_" + str(data_shape), epoch, \
                        data_shape, mean_pixels, ctx=ctx)
    return detector


def parse_args():
    parser = argparse.ArgumentParser(description='Tool to annotate images')
    # Necessary for annotation
    parser.add_argument('--dir', dest='dir', default='./data/demo',
                        help='Directory of images', type=str)
    parser.add_argument('--output', dest='output', default='',
                        help='Name of the output dir, rel. to the input dir', type=str)
    parser.add_argument('--pattern', dest='pattern', type=str, default='*.jpg',
                        help='Glob pattern for the image files')
    parser.add_argument('--batch', dest='batch', type=float, default=100,
                        help='Batch size (detection of images at once)')
    parser.add_argument('--classes', dest='classes', type=str,
                        default='berlinerdom,brandenburgertor,rathausberlin,reichstag',
                        help='Classes to detect')
    parser.add_argument('--filter', dest='filter',
                        help='Remove images that werent classified for the mentioned label',
                        default='')
    parser.add_argument('--limit', dest='limit',
                        help='limits the number of annotated images', default=500)

    # Taken from demo.py. Typically can be left untouched
    parser.add_argument('--prefix', dest='prefix', help='trained model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'ssd'), type=str)
    parser.add_argument('--network', dest='network', type=str, default='vgg16_ssd_300',
                        choices=['vgg16_ssd_300', 'vgg16_ssd_512'], help='which network to use')
    parser.add_argument('--epoch', dest='epoch', help='epoch of trained model',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu', help='(override GPU) use CPU to detect',
                        action='store_true', default=False)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0,
                        help='GPU device id to detect with')
    parser.add_argument('--thresh', dest='thresh', type=float, default=0.5,
                        help='object visualize score threshold, default 0.5')
    parser.add_argument('--mean-r', dest='mean_r', type=float, default=123,
                        help='red mean value')
    parser.add_argument('--mean-g', dest='mean_g', type=float, default=117,
                        help='green mean value')
    parser.add_argument('--mean-b', dest='mean_b', type=float, default=104,
                        help='blue mean value')
    parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.5,
                        help='non-maximum suppression threshold, default 0.5')
    parser.add_argument('--force', dest='force_nms', type=bool, default=True,
                        help='force non-maximum suppression on different class')
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=300,
                        help='set image shape')
    return parser.parse_args()


def annotate(detector, annotations_folder, imagelist, threshold, batchsize, classes):
    batches = int(math.ceil(len(imagelist) / float(batchsize)))
    for i in range(0, batches):
        annotate_batch(detector, annotations_folder,
                       imagelist[i * batchsize:min((i + 1) * batchsize, len(image_list))],
                       threshold, classes)


def annotate_batch(detector, annotations_folder, imagelist, threshold, classes):
    dets = detector.im_detect(imagelist, "", "", True)
    assert len(dets) == len(imagelist)
    success = 0
    for k, det in enumerate(dets):
        success += store_annotation(annotations_folder, imagelist[k], det, threshold, classes)
    print "Found annotations for ", success, "of", len(imagelist), "images"


def should_filter(label):
    global args
    if args.filter != '' and label != args.filter:
        print 'Skipping label', label
        return True
    return False

def store_annotation(annotations_folder, image, dets, threshold, classes):
    best_detection = get_best_detection(dets, threshold, image, classes)
    annotation_file = os.path.join(annotations_folder, best_detection['id'] + '.json')
    with open(annotation_file, 'w') as output:
        json.dump(best_detection, output)
    return 1 if 'label' in best_detection else 0


def filename_wo_ext(path):
    base = os.path.basename(path)
    ext = os.path.splitext(path)[1]
    return base[0:-len(ext)]


def get_best_detection(dets, thresh, image_path, classes):
    detection = {
        "id": filename_wo_ext(image_path),
        "score": 0,
        "annotationStatus": "none"
    }
    for i in range(dets.shape[0]):
        cls_id = int(dets[i, 0])
        if cls_id >= 0:
            score = float(dets[i, 1])
            if score > detection['score'] and score > thresh:
                xmin = dets[i, 2]
                ymin = dets[i, 3]
                xmax = dets[i, 4]
                ymax = dets[i, 5]
                detection['score'] = score
                detection['boundingBox'] = {
                    "x": float(xmin),
                    "y": float(ymin),
                    "width": float(xmax - xmin),
                    "height": float(ymax - ymin)
                }
                class_name = str(cls_id)
                if classes and len(classes) > cls_id:
                    class_name = classes[cls_id]
                detection['label'] = str(class_name)
                detection['annotationStatus'] = 'autoAnnotated'
    return detection

def filter_images(folder, image_list, limit):
    to_annotate = []
    for image in image_list:
        if len(to_annotate) >= limit:
            return to_annotate
        id = filename_wo_ext(image)
        annotation_path = os.path.join(folder, id+".json")
        if os.path.exists(annotation_path):
            with open(annotation_path) as data_file:
                data = json.load(data_file)
                if 'annotationStatus' not in data or data['annotationStatus'] == 'none':
                    to_annotate.append(image)
        else:
            to_annotate.append(image)
    return to_annotate

if __name__ == '__main__':
    args = parse_args()
    if args.cpu:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu_id)

    annotations_path = os.path.join(args.dir, args.output)

    if not os.path.exists(annotations_path):
        os.mkdir(annotations_path)

    image_glob = os.path.join(args.dir, args.pattern)
    image_list = filter_images(args.dir, glob.glob(image_glob), int(args.limit))

    assert len(image_list) > 0, "No valid image specified to detect"
    network = args.network
    detector = get_detector(network, args.prefix, args.epoch,
                            args.data_shape,
                            (args.mean_r, args.mean_g, args.mean_b),
                            ctx,
                            args.classes.split(","),
                            args.nms_thresh, args.force_nms)
    annotate(detector, annotations_path, image_list, args.thresh, int(args.batch),
             args.classes.split(","))
