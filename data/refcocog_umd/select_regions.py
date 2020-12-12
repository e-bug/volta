import json
import argparse
from collections import defaultdict


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', default='res101_coco_minus_refer_notime_dets.json', help='Input filepath')
    parser.add_argument('--outfile', default='res101_coco_minus_refer_notime_dets_36.json', help='Output filepath')
    parser.add_argument('--max_regions', default=36, type=int, help='Maximum number of image regions')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # Load detections
    j = json.load(open(args.infile))
    img2tuple = defaultdict(list)
    for d in j:
        img2tuple[d['image_id']].append((d['det_id'], d['score']))

    # Retrieve top-N
    img2tupleN = defaultdict(list)
    for img, t in img2tuple.items():
        if len(t) <= args.max_regions:
            img2tupleN[img] = t
        else:
            img2tupleN[img] = sorted(img2tuple[img], key=lambda x: x[1], reverse=True)[:args.max_regions]
    det_ids = {t[0] for img in img2tupleN for t in img2tupleN[img]}
    jN = [d for d in j if d['det_id'] in det_ids]

    # Store top-N detections
    with open(args.outfile, 'w') as f:
        json.dump(jN, f)
