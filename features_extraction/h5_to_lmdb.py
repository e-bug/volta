import lmdb
import h5py
from tqdm import tqdm
import pickle
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', type=str)
    parser.add_argument('--lmdb', type=str)
    args = parser.parse_args()

    source_fname = args.h5
    target_fname = args.lmdb
    env = lmdb.open(target_fname, map_size=1099511627776, writemap=True)
    f = h5py.File(source_fname, 'r')
    print(f'Writing {target_fname} from {source_fname}')
    with env.begin(write=True) as txn:
        id_list = []
        for i, img_id in enumerate(tqdm(f.keys(), ncols=100)):
            if i == 0:
                keys = list(f[img_id].keys())

            item = {}
            for k in keys:
                item[k] = f[f'{img_id}/{k}'][()]
            item['img_id'] = img_id

            txn.put(img_id.encode(), pickle.dumps(item))
            id_list.append(img_id.encode())

        txn.put('keys'.encode(), pickle.dumps(id_list))
    f.close()
