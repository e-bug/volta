#!/bin/bash

source /home/projects/ku_00062/envs/iglue/bin/activate

cd ../..
for split in few test; do
  INDIR=/home/projects/ku_00062/data/xFlickrCO/features/xflickrco-${split}_X101.npy
  OUTDIR=/home/projects/ku_00062/data/xFlickrCO/features/xflickrco-${split}_X101.lmdb
  
  python npy_to_lmdb.py --mode convert --features_folder $INDIR --lmdb_path $OUTDIR

done

deactivate
