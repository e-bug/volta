# !/usr/bin/env python

import sys
import csv
import argparse
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features",
              "cls_prob", "attrs", "classes"]

# Settings for the number of features per image. To re-create pretrained features with 36 features
# per image, set both values to 36.
MIN_BOXES = 36
MAX_BOXES = 36


def merge_tsvs(fname, total_group, wanted_ids):
    fnames = ['%s.%d' % (fname, i) for i in range(total_group)]

    outfile = fname
    with open(outfile, 'ab') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
        found_ids = set()
        for infile in fnames:
            with open(infile) as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
                for item in tqdm(reader):
                    img_id = int(item['img_id'])
                    if (img_id not in found_ids) and (img_id in wanted_ids):
                        try:
                            writer.writerow(item)
                            found_ids.add(img_id)
                        except Exception as e:
                            print(e)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--split', type=str, default='valid')
    parser.add_argument('--total_group', type=int, default=1,
                        help="the number of group for extracting")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Setup the configuration, normally do not need to touch these:
    args = parse_args()
    args.outfile = "%s_obj%d-%d.tsv" % (args.split, MIN_BOXES, MAX_BOXES)

    # Get clean ids
    clean_ids = set()
    for fn in ['train_ids.txt', 'valid_ids.txt']:
        with open(fn) as f:
            for l in f.readlines():
                clean_ids.add(int(l.strip()))

    # Generate TSV files, normally do not need to modify
    merge_tsvs(args.outfile, args.total_group, clean_ids)
