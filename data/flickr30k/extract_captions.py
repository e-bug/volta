import os
import json

import argparse
from tqdm import tqdm
import jsonlines


SPLIT2NAME = {
    'train': 'train',
    'valid': 'val',
    'test': 'test',
}


def get_annotations(infile, save_path, split):

    lst = json.load(open(infile))['images']
    split_list = [e for e in lst if e['split'] == split]

    with jsonlines.open(save_path, mode='w') as writer:
        for e in tqdm(split_list):
            sentences = [d['raw'] for d in e['sentences']]
            name = e['filename']
            img_id = name.split('.')[0]
            d = {'sentences': sentences, 'id': img_id, 'img_path': name}
            writer.write(d)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Convert to LMDB')
    parser.add_argument('--infile', type=str, default='data/flickr30k/dataset_flickr30k.json')
    parser.add_argument('--outdir', type=str, default='data/flickr30k/annotations')
    parser.add_argument('--split', type=str, default='valid', choices=['train', 'valid', 'test'])

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Setup the configuration, normally do not need to touch these:
    args = parse_args()
    args.outfile = os.path.join(args.outdir, "%s_ann.jsonl" % args.split)
    
    print('Called with args:')
    print(args)

    # Extract annotations
    get_annotations(args.infile, args.outfile, SPLIT2NAME[args.split])
