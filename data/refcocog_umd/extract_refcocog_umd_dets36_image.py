# !/usr/bin/env python

# The root of bottom-up-attention repo. Do not need to change if using provided docker file.
BUTD_ROOT = '/opt/butd/'

import os, sys
sys.path.insert(0, BUTD_ROOT + "/tools")
os.environ['GLOG_minloglevel'] = '2'

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from fast_rcnn.test import im_detect, _get_blobs
from fast_rcnn.nms_wrapper import nms

import caffe
import argparse
import pprint
import base64
import numpy as np
import cv2
import csv
from tqdm import tqdm
import json

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features",
              "cls_prob", "attrs", "classes"]

# Settings for the number of features per image. To re-create pretrained features with 36 features
# per image, set both values to 36.
MIN_BOXES = 36
MAX_BOXES = 36


def load_image_ids(img_root, group_id, total_group, detections_path):
    """images in the same directory are in the same split"""
    image_data = {}
    image_id = {}
    with open(detections_path) as f:
        for item in json.load(f):
            if item['image_id'] not in image_data:
                image_data[item['image_id']] = []
            image_data[item['image_id']].append(item)

    fnames = ['COCO_train2014_%012d.jpg' % int(im_id) for im_id in image_data.keys()]
    img_ids = [int(im_id) for im_id in image_data]
    imgs = list(image_data.values())

    total_num = len(fnames)
    per_num = int(np.ceil(total_num / total_group))
    if group_id == total_group - 1:
        current_lst = fnames[int(group_id * per_num):]
        current_ids = img_ids[int(group_id * per_num):]
        current_imgs = imgs[int(group_id * per_num):]
    else:
        current_lst = fnames[int(group_id * per_num):int((group_id+1)*per_num)]
        current_ids = img_ids[int(group_id * per_num):int((group_id+1)*per_num)]
        current_imgs = imgs[int(group_id * per_num):int((group_id+1)*per_num)]
    print('current: %d-%d/%d' % (int(group_id * per_num), int(group_id * per_num)+len(current_lst), total_num))

    max_num_box = max([len(box) for box in image_data.values()])
    bbox = np.zeros([len(image_data), max_num_box, 5])
    num_bbox = np.zeros(len(image_data))

    pathXid = []
    count = 0
    for ix, name in enumerate(current_lst):
        num = 0
        for i, image in enumerate(current_imgs[ix]):
            bbox[count, i, 1] = image['box'][0]
            bbox[count, i, 2] = image['box'][1]
            bbox[count, i, 3] = image['box'][0] + image['box'][2]
            bbox[count, i, 4] = image['box'][1] + image['box'][3]
            num += 1
        num_bbox[count] = num
        count += 1

        idx = current_ids[ix]
        filepath = os.path.join(img_root, name)
        pathXid.append((filepath, idx))

    return pathXid, bbox, num_bbox


def generate_tsv(prototxt, weights, image_ids, bbox, num_bbox, outfile):
    # First check if file exists, and if it is complete
    wanted_ids = set([image_id[1] for image_id in image_ids])
    found_ids = set()
    if os.path.exists(outfile):
        with open(outfile, "r") as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
            for item in reader:
                found_ids.add(item['img_id'])
    missing = wanted_ids - found_ids
    if len(missing) == 0:
        print('already completed {:d}'.format(len(image_ids)))
    else:
        print('missing {:d}/{:d}'.format(len(missing), len(image_ids)))
    if len(missing) > 0:
        caffe.set_mode_gpu()
        caffe.set_device(0)
        net = caffe.Net(prototxt, caffe.TEST, weights=weights)
        with open(outfile, 'ab') as tsvfile:
            writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
            for ii, image in tqdm(enumerate(image_ids)):
                im_file, image_id = image
                if image_id in missing:
                    try:
                        row = get_detections_from_im(net, im_file, image_id, bbox[ii], int(num_bbox[ii]))
                        if row is not None:
                            writer.writerow(row)
                    except Exception as e:
                        print(e)


def get_detections_from_im(net, im_file, image_id, bbox=None, num_bbox=None, conf_thresh=0.2):
    """
    :param net:
    :param im_file: full path to an image
    :param image_id:
    :param conf_thresh:
    :return: all information from detection and attr prediction
    """
    im = cv2.imread(im_file)
    if bbox is not None:
        scores, boxes, attr_scores, rel_scores = im_detect(net, im, bbox[:num_bbox, 1:], force_boxes=True)
    else:
        scores, boxes, attr_scores, rel_scores = im_detect(net, im)

    # Keep the original boxes, don't worry about the regression bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data
    attr_prob = net.blobs['attr_prob'].data
    pool5 = net.blobs['pool5_flat'].data

    objects = np.argmax(cls_prob[:, 1:], axis=1)
    objects_conf = np.max(cls_prob[:, 1:], axis=1)
    attrs = np.argmax(attr_prob[:, 1:], axis=1)
    attrs_conf = np.max(attr_prob[:, 1:], axis=1)

    return {
        "img_id": image_id,
        "img_h": np.size(im, 0),
        "img_w": np.size(im, 1),
        "objects_id": base64.b64encode(objects),  # int64
        "objects_conf": base64.b64encode(objects_conf),  # float32
        "attrs_id": base64.b64encode(attrs),  # int64
        "attrs_conf": base64.b64encode(attrs_conf),  # float32
        "num_boxes": len(rois),
        "boxes": base64.b64encode(cls_boxes),  # float32
        "features": base64.b64encode(pool5),  # float32
        "cls_prob": base64.b64encode(cls_prob),
        "classes": base64.b64encode(scores),
        "attrs": base64.b64encode(attr_scores)
    }


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id(s) to use',
                        default='0', type=str)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--out', dest='outfile',
                        help='output filepath',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--imgroot', type=str, default='/workspace/images/')
    parser.add_argument('--caffemodel', type=str, default='../snap/pretrained/resnet101_faster_rcnn_final_iter_320000.caffemodel')
    parser.add_argument('--total_group', type=int, default=1,
                        help="the number of group for extracting")
    parser.add_argument('--group_id', type=int, default=0,
                        help=" group id for current analysis, used to shard")
    parser.add_argument('--detections', type=str, default='../detections/res101_coco_minus_refer_notime_dets_36.json')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Setup the configuration, normally do not need to touch these:
    args = parse_args()

    args.cfg_file = BUTD_ROOT + "experiments/cfgs/faster_rcnn_end2end_resnet.yml"
    args.prototxt = BUTD_ROOT + "models/vg/ResNet-101/faster_rcnn_end2end_final/test_gt.prototxt"
    args.outfile = "%s_obj%d-%d.tsv.%d" % ('refcocog_umd_dets36', MIN_BOXES, MAX_BOXES, args.group_id)
    
    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)
    assert cfg.TEST.HAS_RPN

    # Load image ids, need modification for new datasets.
    image_ids, bbox, num_bbox = load_image_ids(args.imgroot, args.group_id, args.total_group, args.detections)
    
    # Generate TSV files, normally do not need to modify
    generate_tsv(args.prototxt, args.caffemodel, image_ids, bbox, num_bbox, args.outfile)
