# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import glob
import os
import pickle

import lmdb
import numpy as np
import tqdm

import base64
from tensorpack.dataflow import RNGDataFlow, PrefetchDataZMQ, LMDBSerializer


class PretrainData(RNGDataFlow):
    def __init__(self, corpus_path, ids_fname, shuffle=False, num_imgs=None):
        self.corpus_path = corpus_path
        self.shuffle = shuffle

        self.features_folders = glob.glob(os.path.join(corpus_path, "*resnext101"))
        self.ids = [l.strip() for l in open(ids_fname).readlines()]

        assert num_imgs is not None
        self.num_imgs = num_imgs
        self.cnt = 0

    def __len__(self):
        return self.num_imgs

    def __iter__(self):
        for p in tqdm.tqdm(self.features_folders, total=len(self.features_folders)):
            all_features = glob.glob(os.path.join(p, "*.npy"))
            features = []
            for feature in all_features:
                if not feature.endswith("_info.npy"):
                    features.append(feature)

            for infile in features:
                if self.cnt == self.num_imgs:
                    break
                
                img_id = infile.split("/")[-1].split(".npy")[0]
                if img_id not in self.ids:
                    continue

                reader = np.load(infile, allow_pickle=True)

                item = {}
                item['img_id'] = img_id
                item["features"] = reader

                info_file = infile.split(".npy")[0] + "_info.npy"

                reader = np.load(info_file, allow_pickle=True)

                item["img_h"] = reader.item().get("image_height")
                item["img_w"] = reader.item().get("image_width")
                item["num_boxes"] = reader.item().get("num_boxes")
                item["objects"] = reader.item().get("objects")
                item["cls_prob"] = reader.item().get("cls_prob", None)
                item["boxes"] = reader.item().get("bbox")

                item["features"] = base64.b64encode(item["features"])
                item["boxes"] = base64.b64encode(item["boxes"])

                self.cnt += 1
                yield item


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_dir', type=str)
    parser.add_argument('--lmdb', type=str)
    parser.add_argument('--ids_fname', type=str)
    parser.add_argument('--num_imgs', type=int, default=None)
    args = parser.parse_args()

    source_path = args.features_dir
    target_fname = args.lmdb

    ds = PretrainData(source_path, args.ids_fname, num_imgs=args.num_imgs)
    ds1 = PrefetchDataZMQ(ds, nr_proc=1)
    LMDBSerializer.save(ds1, target_fname)
