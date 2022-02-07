import h5py
import argparse
from tensorpack.dataflow import RNGDataFlow, PrefetchDataZMQ, LMDBSerializer


class PretrainData(RNGDataFlow):
    def __init__(self, corpus_path, shuffle=False, num_imgs=None):
        self.corpus_path = corpus_path
        self.shuffle = shuffle

        if num_imgs is None:
            with h5py.File(corpus_path, 'r') as f:
                num_imgs = len(f)
        self.num_imgs = num_imgs

    def __len__(self):
        return self.num_imgs

    def __iter__(self):
        with h5py.File(self.corpus_path, 'r') as f:
            for i, img_id in enumerate(f.keys()):
                if i == 0:
                    keys = list(f[img_id].keys())

                item = {}
                for k in keys:
                    item[k] = f[f'{img_id}/{k}'][()]
                item['img_id'] = img_id

                yield item


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', type=str)
    parser.add_argument('--lmdb', type=str)
    parser.add_argument('--num_imgs', type=int, default=None)
    args = parser.parse_args()

    source_fname = args.h5
    target_fname = args.lmdb

    ds = PretrainData(source_fname, num_imgs=args.num_imgs)
    ds1 = PrefetchDataZMQ(ds, nr_proc=1)
    LMDBSerializer.save(ds1, target_fname)
