#!/bin/bash

DATA="/home/projects/ku_00062/data/marvl/features"

source /home/projects/ku_00062/envs/iglue/bin/activate

cd ../..
for lang in id sw ta tr zh; do
  H5="${DATA}/marvl-${lang}_boxes36.h5"
  LMDB="${DATA}/marvl-${lang}_boxes36.lmdb"

  python h5_to_lmdb.py --h5 $H5 --lmdb $LMDB
done

deactivate
