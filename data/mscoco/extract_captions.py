import os
import json
import argparse
import jsonlines
from tqdm import tqdm

SPLIT2NAME = {
    'train': 'train',
    'valid': 'val',
    'test': 'test',
    'test1k': 'test1k',
}


def get_annotations(infile, save_path, split):

    lst = json.load(open(infile))['images']

    if split == 'test1k':
        ids = set([int(i.strip()) for i in open('test1k_ids.txt').readlines()])
        split_list = [e for e in lst if e['cocoid'] in ids]
    else:
        split_list = [e for e in lst if e['split'] == split]
    if split == 'train':
        split_list += [e for e in lst if e['split'] == 'restval']

    with jsonlines.open(save_path, mode='w') as writer:
        for e in tqdm(split_list):
            sentences = [d['raw'] for d in e['sentences']]
            img_id = e['cocoid']
            d = {'sentences': sentences, 'id': img_id}
            writer.write(d)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Convert to LMDB')
    parser.add_argument('--infile', type=str, default='data/mscoco/dataset_coco.json')
    parser.add_argument('--outdir', type=str, default='data/mscoco/annotations')
    parser.add_argument('--split', type=str, default='valid', choices=['train', 'valid', 'test', 'test1k'])

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

