# coding=utf-8
# Copyleft 2019 Project LXRT

import sys
import csv
import base64
import time
from tqdm import tqdm
import numpy as np
import h5py
import argparse

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv_path', type=str,
                        default='val2014_obj36.tsv')
    parser.add_argument('--h5_path', type=str,
                        default='val2014_obj36.h5')

    args = parser.parse_args()
    dim = 2048

    print('Load ', args.tsv_path)
    output_fname = args.h5_path
    print('features will be saved at', output_fname)
    
    with h5py.File(output_fname, 'w') as f:
        with open(args.tsv_path) as inf:
            reader = csv.DictReader(inf, FIELDNAMES, delimiter="\t")
            for i, item in tqdm(enumerate(reader), ncols=150):

                img_id = item['img_id']

                for key in ['img_h', 'img_w', 'num_boxes']:
                    item[key] = int(item[key])

                num_boxes = item['num_boxes']
                decode_config = [
                    ('objects_id', (num_boxes, ), np.int64),
                    ('objects_conf', (num_boxes, ), np.float32),
                    ('attrs_id', (num_boxes, ), np.int64),
                    ('attrs_conf', (num_boxes, ), np.float32),
                    ('boxes', (num_boxes, 4), np.float32),
                    ('features', (num_boxes, -1), np.float32),
                ]
                for key, shape, dtype in decode_config:
                    item[key] = np.frombuffer(
                        base64.b64decode(item[key]), dtype=dtype)
                    item[key] = item[key].reshape(shape)
                    item[key].setflags(write=False)

                grp = f.create_group(img_id)
                grp['features'] = item['features'].reshape(num_boxes, 2048)
                grp['obj_id'] = item['objects_id']
                grp['obj_conf'] = item['objects_conf']
                grp['attr_id'] = item['attrs_id']
                grp['attr_conf'] = item['attrs_conf']
                grp['boxes'] = item['boxes']
                grp['img_w'] = item['img_w']
                grp['img_h'] = item['img_h']
