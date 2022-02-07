#!/bin/bash

INDIR=/home/projects/ku_00062/data/gqa/features/vg-gqa_X101.npy
OUTDIR=/home/projects/ku_00062/data/gqa/features/vg-gqa_X101.lmdb

mkdir -p $OUTDIR

source /home/projects/ku_00062/envs/iglue/bin/activate

python npy_to_lmdb.py --mode convert --features_folder $INDIR --lmdb_path $OUTDIR

deactivate
