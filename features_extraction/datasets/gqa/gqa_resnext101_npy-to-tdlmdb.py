# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import glob
import os
import pickle

import lmdb
import numpy as np
import tqdm

import base64
from collections import defaultdict
from tensorpack.dataflow import RNGDataFlow, PrefetchDataZMQ, LMDBSerializer


class PretrainData(RNGDataFlow):
    def __init__(self, corpus_path, entries, shuffle=False):
        self.corpus_path = corpus_path
        self.shuffle = shuffle

        self.features_folders = [corpus_path]

        self.img2entries = defaultdict(list)
        for e in entries:
            e['image_id'] = str(e['image_id'])
            self.img2entries[e['image_id']].append(e)
        self.ids = set(self.img2entries.keys())

        self.num_imgs = len(self.img2entries)
        self.num_entries = len(entries)

        self.cnt = 0

    def __len__(self):
        return self.num_entries

    def __iter__(self):
        for p in self.features_folders:
            features = [os.path.join(p, f"{img_id}.npy") for img_id in self.img2entries.keys()]

            for infile in features:
                
                img_id = infile.split("/")[-1].split(".npy")[0]

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

                for e in self.img2entries[img_id]:
                    item["entry"] = e

                    self.cnt += 1
                    yield item


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_dir', type=str)
    parser.add_argument('--lmdb', type=str)
    parser.add_argument('--annotation', type=str)
    args = parser.parse_args()

    source_path = args.features_dir
    target_fname = args.lmdb
    entries = pickle.load(open(args.annotation, "rb"))

    ds = PretrainData(source_path, entries) 
    ds1 = PrefetchDataZMQ(ds, nr_proc=1)
    LMDBSerializer.save(ds1, target_fname)
