# coding=utf-8
import sys
sys.path.append("../..")

from detectron2_proposal_maxnms import collate_fn, extract, NUM_OBJECTS, DIM
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm
from pathlib import Path
import argparse
import os.path
import glob


class ImageDataset(Dataset):
    def __init__(self, image_dirs):
        self.image_dirs = image_dirs
        self.image_path_list = []
        for img_dir in image_dirs:
            self.image_path_list.extend(glob.glob(os.path.join(img_dir, "*")))
        self.n_images = len(self.image_path_list)

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        image_id = image_path.split("/")[-1]

        img = cv2.imread(image_path)

        return {
            'img_id': image_id,
            'img': img
        }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=1, type=int, help='batch_size')
    parser.add_argument('--root', type=str, default='datasets/flickr30k/')
    parser.add_argument('--split', type=str, default=None, choices=['test', 'few'])
    parser.add_argument('--outdir', type=str)
    args = parser.parse_args()

    dataset_name = 'xflickr'

    out_dir = args.outdir

    dataset = ImageDataset(os.path.join(args.root, args.split))

    dataloader = DataLoader(dataset, batch_size=args.batchsize,
                            shuffle=False, collate_fn=collate_fn, num_workers=4)

    output_fname = os.path.join(out_dir, f'{dataset_name}-{args.split}_boxes{NUM_OBJECTS}.h5')
    print('features will be saved at', output_fname)

    desc = f'{dataset_name}_{args.split}_{(NUM_OBJECTS, DIM)}'

    extract(output_fname, dataloader, desc)
