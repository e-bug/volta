import os
import glob
import tqdm
import h5py
import argparse
from tensorpack.dataflow import RNGDataFlow, PrefetchDataZMQ, LMDBSerializer


class PretrainData(RNGDataFlow):
    def __init__(self, corpus_path, ids_fname, shuffle=False, num_imgs=None):
        self.corpus_path = corpus_path
        self.shuffle = shuffle

        self.fnames = glob.glob(os.path.join(corpus_path, "*.h5"))
        self.ids = [l.strip() for l in open(ids_fname).readlines()]

        assert num_imgs is not None

        self.num_imgs = num_imgs
        self.cnt = 0

    def __len__(self):
        return self.num_imgs

    def __iter__(self):
        for fn in tqdm.tqdm(self.fnames, total=len(self.fnames)):
            with h5py.File(fn, 'r') as f:
                for i, img_id in enumerate(f.keys()):
                    if i == 0:
                        keys = list(f[img_id].keys())
                    if self.cnt == self.num_imgs:
                        break
                    if img_id not in self.ids:
                        continue

                    item = {}
                    for k in keys:
                        item[k] = f[f'{img_id}/{k}'][()]
                    item['img_id'] = img_id

                    self.cnt += 1

                    yield item


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_dir', type=str)
    parser.add_argument('--lmdb', type=str)
    parser.add_argument('--ids_fname', type=str)
    parser.add_argument('--num_imgs', type=int, default=None)
    args = parser.parse_args()

    source_path = args.h5_dir
    target_fname = args.lmdb

    ds = PretrainData(source_path, args.ids_fname, num_imgs=args.num_imgs)
    ds1 = PrefetchDataZMQ(ds, nr_proc=1)
    LMDBSerializer.save(ds1, target_fname)
