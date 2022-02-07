#!/bin/bash

basedir=/home/projects/ku_00062

INDIR="${basedir}/data/gqa/features/vg-gqa_X101.npy"
OUTDIR="${basedir}/data/gqa/features/gqa-train_X101.lmdb"
TEXT="${basedir}/data/gqa/annotations/train_target.pkl"
rm -r $OUTDIR

source ${basedir}/envs/iglue/bin/activate

python gqa_resnext101_npy-to-tdlmdb.py --features_dir $INDIR --lmdb $OUTDIR --annotation $TEXT

deactivate
