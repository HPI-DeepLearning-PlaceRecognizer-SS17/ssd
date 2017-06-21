from __future__ import print_function
import argparse
import glob

import os
import json

def filename_wo_ext(path):
    base = os.path.basename(path)
    ext = os.path.splitext(path)[1]
    return base[0:-len(ext)]

def process_json(json_path, source_status, target_status, label):
    with open(json_path, 'r+') as data_file:
        try:
            annotation = json.load(data_file)
        except ValueError as err:
            print(err)
            annotation = {
                "id": filename_wo_ext(json_path)
            }
       # print(annotation["annotationStatus"], annotation["annotationStatus"] == source_status)
        if "annotationStatus" not in annotation or annotation["annotationStatus"] == source_status:
            annotation["boundingBox"] = {
                "x": 0,
                "y": 0,
                "width": 1,
                "height": 1
            }
            annotation["label"] = label
            annotation["annotationStatus"] = target_status
            data_file.seek(0)
            json.dump(annotation, data_file)
            data_file.truncate()
            return 1
    return 0



def process_images(dir, label, source_status, target_status, image_list):
    annotated = 0
    for image_path in image_list:
        image_id = filename_wo_ext(image_path)
        json_path = os.path.join(dir, image_id+".json")
        annotated += process_json(json_path, source_status, target_status, label)
    print("Annotated", annotated, "Images")



def parse_args():
    parser = argparse.ArgumentParser(description='Tool to annotate images')
    # Necessary for annotation
    parser.add_argument('--dir', dest='dir', required=True,
                        help='Directory of images & annotations', type=str)
    parser.add_argument('--label', dest='label',
                        help='The label we want to annotate them with',
                        default='none')
    parser.add_argument('--source-status', dest='source_status',
                        help='The annotation status of images that will be processed',
                        default='none')
    parser.add_argument('--target-status', dest='target_status',
                        help='The annotation status they will receive afterwards',
                        default='autoAnnotated')
    parser.add_argument('--pattern', dest='pattern', type=str, default='*.jpg',
                        help='Glob pattern for the image files')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    image_glob = os.path.join(args.dir, args.pattern)
    image_list = glob.glob(image_glob)
    process_images(args.dir, args.label, args.source_status, args.target_status, image_list)