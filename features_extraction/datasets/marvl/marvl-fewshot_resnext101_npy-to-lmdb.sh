#!/bin/bash

source /home/projects/ku_00062/envs/iglue/bin/activate

cd ../..
for lang in id sw ta tr zh; do
  INDIR=/home/projects/ku_00062/data/marvl/few_shot/features/marvl-${lang}_X101.npy
  OUTDIR=/home/projects/ku_00062/data/marvl/few_shot/features/marvl-${lang}_X101.lmdb
  
  python npy_to_lmdb.py --mode convert --features_folder $INDIR --lmdb_path $OUTDIR

done

deactivate
