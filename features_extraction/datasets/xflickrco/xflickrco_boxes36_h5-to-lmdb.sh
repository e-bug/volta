#!/bin/bash

DATA="/home/projects/ku_00062/data/xFlickrCO/features"

source /home/projects/ku_00062/envs/iglue/bin/activate

cd ../..
for split in test few; do
  H5="${DATA}/xflickrco-${split}_boxes36.h5"
  LMDB="${DATA}/xflickrco-${split}_boxes36.lmdb"

  python h5_to_lmdb.py --h5 $H5 --lmdb $LMDB
done

deactivate
