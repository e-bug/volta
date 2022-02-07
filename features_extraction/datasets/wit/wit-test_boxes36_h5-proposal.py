# coding=utf-8
import sys
sys.path.append("../..")

from detectron2_proposal_maxnms import collate_fn, extract, NUM_OBJECTS, DIM
from torch.utils.data import Dataset, DataLoader
import cv2
import base64
import os.path
import argparse
import numpy as np
import pandas as pd


class ImageDataset(Dataset):
    def __init__(self, images):
        self.images = images
        self.n_images = len(self.images)


    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        image_id = 'wit-test_%d' % idx

        im_b64 = self.images[idx]
        im_bytes = base64.b64decode(im_b64)
        im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
        img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

        return {
            'img_id': image_id,
            'img': img
        }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=1, type=int, help='batch_size')
    parser.add_argument('--root', type=str, default='datasets/flickr30k/')

    args = parser.parse_args()

    dataset_name = 'wit'

    print('Loading images of WIT')
    df = pd.read_csv(os.path.join(args.root, 'chosen_langs.tsv'), sep='\t', usecols=['base64_image'])
    print('# Images:', len(df))

    dataset = ImageDataset(df['base64_image'].values)

    dataloader = DataLoader(dataset, batch_size=args.batchsize,
                            shuffle=False, collate_fn=collate_fn, num_workers=4)

    output_fname = os.path.join(args.root, f'wit-test_boxes{NUM_OBJECTS}.h5')
    print('features will be saved at', output_fname)

    desc = f'{dataset_name}_{(NUM_OBJECTS, DIM)}'

    extract(output_fname, dataloader, desc)
